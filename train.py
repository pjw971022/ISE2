import os 
import sys
import math
import time
import shutil
import pickle
from lockfile import LockFile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.cuda.amp
import numpy as np
import wandb

import data
# import data_cub
from vocab import Vocabulary
from model import PVSE
from loss import PVSELoss, ChamferLoss
from loss import AsymmetricTripletLoss
from eval import i2t, t2i, encode_data
# from eval_cub import i2t_cub, t2i_cub
from logger import AverageMeter
from option import parser, verify_input_args
from sync_batchnorm import convert_model, SynchronizedBatchNorm2d
from distance import SetwiseDistance
from model_perceiver import PVSE_perceiver
from warmup_scheduler import GradualWarmupScheduler

import adamp

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
total_iter = 0

def lock_and_write_to_file(filename, text):
    with LockFile(filename) as lock:
        with open(filename, 'a') as fid:
            fid.write('{}\n'.format(text))


def copy_input_args_from_ckpt(args, ckpt_args):
    args_to_copy = ['word_dim','crop_size','cnn_type','embed_size', 'num_embeds',
                                    'img_attention','txt_attention','max_video_length']
    for arg in args_to_copy:
        val1, val2 = getattr(args, arg), getattr(ckpt_args, arg)
        if val1 != val2:
            logging.warning('Updating argument from checkpoint [{}]: [{}] --> [{}]'.format(arg, val1, val2))
            setattr(args, arg, val2)
    return args

def save_ckpt(state, is_best, filename='ckpt.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        logging.info('Updating the best model checkpoint: {}'.format(prefix + 'model_best.pth.tar'))


def get_description(args, epoch=-1):
    return ('[{}][epoch:{}] {}'.format(args.logger_name.split('/')[-1], epoch, args))


def train(epoch, data_loader, model, criterion, optimizer, scaler, args, lr_warmup=False, scheduler=None):
    global total_iter
    # switch to train mode
    model.train()
    if args.bn_eval:
        modules = model.module.modules() if args.multi_gpu else model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
    
    # debug_criterion = ChamferLoss(args, smooth=True)
    # average meters to record the training statistics
    losses = AverageMeter()
    stat_dict = dict()
    stat_dict['img_vars'] = AverageMeter()
    stat_dict['txt_vars'] = AverageMeter()
    stat_dict['total_vars'] = AverageMeter()
    losses_dict = dict()
    losses_dict['ranking_loss'] = AverageMeter()
    losses_dict['i2t_loss'] = AverageMeter()
    losses_dict['t2i_loss'] = AverageMeter()
    if args.div_weight > 0:
        losses_dict['div_loss'] = AverageMeter()
    if args.mmd_weight > 0:
        losses_dict['mmd_loss'] = AverageMeter()
    if args.unif_weight > 0:
        losses_dict['unif_loss'] = AverageMeter()
    if args.qreg_weight > 0:
        losses_dict['query_regularizer'] = AverageMeter()

    total_batches = len(data_loader)# coco 200 * 567
    print('##training##')
    for itr, data in enumerate(data_loader):
        total_iter += 1
        if torch.cuda.is_available():
            if 'butd' in args.data_name:
                if args.fast_batch:
                    img, txt, img_len, txt_len, recovery, _ = data
                    img, txt, img_len, txt_len, recovery = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda(), recovery.cuda()
                else:
                    img, txt, img_len, txt_len, _ = data
                    img, txt, img_len, txt_len = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda()
            else:
                img_len = None
                if args.fast_batch:
                    img, txt, txt_len, recovery, _ = data
                    img, txt, txt_len, recovery = img.cuda(), txt.cuda(), txt_len.cuda(), recovery.cuda()
                else:
                    img, txt, txt_len, _ = data
                    img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()
        else: 
            assert False
            
        with torch.cuda.amp.autocast(enabled=args.amp):
            # Forward pass and compute loss; _a: attention map, _r: residuals
            img_emb, txt_emb, img_attn, txt_attn, img_r, txt_r = model.forward(img, txt, img_len, txt_len)
            # Compute loss and update statstics. Give loss a recovery label when args.fast_batch is on.
            if args.fast_batch:
                txt_emb = txt_emb[recovery]
                txt_r = txt_r[recovery]
            
            if args.qreg_weight > 0:
                loss, loss_dict = criterion(img_emb, txt_emb, img_r, txt_r, \
                    img_query=model.img_enc.pie_net.perceiver.latents, txt_query=model.txt_enc.pie_net.perceiver.latents)
            else:
                loss, loss_dict = criterion(img_emb, txt_emb, img_r, txt_r)

            if total_iter < lr_warmup:
                loss *= float(total_iter) / lr_warmup
        
        if torch.isnan(loss).any():
            print("!! NaN loss detected !!")
            import ipdb; ipdb.set_trace()
            
        losses.update(loss)
        for key, val in loss_dict.items():
            losses_dict[key].update(val)

        # Backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        wandb.log({'iter':total_iter})
        if scheduler is not None and not lr_warmup:
            scheduler.step()
        
        # Print log info
        if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
            log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
            for key, val in losses_dict.items():
                log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
            n = int(math.ceil(math.log(len(data_loader) + 1, 10)))
            logging.info('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), log_msg))
        
            with torch.no_grad():
                img_emb = img_emb.flatten(end_dim=1)
                txt_emb = txt_emb.flatten(end_dim=1)
                img_var = 1 - img_emb.mean(dim=0).norm(dim=0, p=2)
                txt_var = 1 - txt_emb.mean(dim=0).norm(dim=0, p=2)
                total_var = 1 - torch.cat((img_emb, txt_emb), dim=0).mean(dim=0).norm(dim=0, p=2)
                stat_dict['img_vars'].update(img_var)
                stat_dict['txt_vars'].update(txt_var)
                stat_dict['total_vars'].update(total_var)
            stat_msg = ''
            for key, val in stat_dict.items():
                stat_msg += '%s:%.4f (%.4f), ' %(key, val.val, val.avg)
            logging.info('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), stat_msg))


    log_msg = 'loss: %.4f' %(losses.avg)
    for key, val in losses_dict.items():
        log_msg += ', %s: %.4f' %(key.replace('_loss',''), val.avg)
    exp_name = args.logger_name.split('/')[-1]
    lock_and_write_to_file(args.log_file, '[%s][%d] %s' %(exp_name, epoch, log_msg))

    del img_emb, txt_emb, img_r, txt_r, loss
    return losses.avg, losses_dict, stat_dict
        

def validate(dataset, data_loader, model, args, distance_fn, validation, epoch=-1, best_score=None):
    # switch to eval mode
    model.eval()

    nreps = 5 if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd'] else 10
    order = args.order if hasattr(args, 'order') and args.order else False

    img_embs, txt_embs = encode_data(model, data_loader, 'butd' in args.data_name, args.eval_on_gpu)
    # 5fold cross-validation, only for MSCOCO
    mean_metrics = None
    if 'coco' in args.data_name and not validation:
        results = []
        for i in range(5):
            r, rt0 = i2t(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                distance_fn,
                nreps=nreps, return_ranks=True, order=args.order, use_gpu=args.eval_on_gpu)
            
            ri, rti0 = t2i(
                img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000], 
                distance_fn,
                nreps=nreps, return_ranks=True, order=args.order, use_gpu=args.eval_on_gpu)
            
            r = (r[0], r[1], r[2], r[3], r[3] / img_embs.shape[0], r[4], r[4] / img_embs.shape[0])
            # print("Image to text: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % r)
            ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / img_embs.shape[0], ri[4], ri[4] / img_embs.shape[0])
            # print("Text to image: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % ri)

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            # print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

        print("-----------------------------------")
        print("Mean metrics from 5-fold evaluation: ")
        print("rsum: %.2f" % (mean_metrics[-1]))
        print("Average i2t Recall: %.2f" % mean_metrics[-3])
        print("Image to text: %.2f %.2f %.2f" % mean_metrics[:3])
        print("Average t2i Recall: %.2f" % mean_metrics[-2])
        print("Text to image: %.2f %.2f %.2f" % mean_metrics[7:10])
    
        recall_1k = (mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[7], mean_metrics[8], mean_metrics[9])
    else:
        recall_1k = (0, 0, 0, 0, 0, 0)
    
    (r1, r5, r10, medr, meanr), (ranks, top1) = i2t(img_embs, txt_embs, distance_fn,
            nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)
    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i) = t2i(img_embs, txt_embs, distance_fn,
            nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

    # sum of recalls to be used for early stopping
    rsum = r1 + r5 + r10 + r1i + r5i + r10i
    med_rsum, mean_rsum = medr + medri, meanr + meanri

    # log
    exp_name = args.logger_name.split('/')[-1]
    vname = 'Video' if args.max_video_length>1 else 'Image'

    log_str1 = "[%s][%d] %s to text: %.2f, %.2f, %.2f, %.2f, %.2f" \
                            %(exp_name, epoch, vname, r1, r5, r10, medr, meanr)
    log_str2 = "[%s][%d] Text to %s: %.2f, %.2f, %.2f, %.2f, %.2f" \
                            %(exp_name, epoch, vname, r1i, r5i, r10i, medri, meanri)
    log_str3 = '[%s][%d] rsum: %.2f, med_rsum: %.2f, mean_rsum: %.2f' \
                            %(exp_name, epoch, rsum, med_rsum, mean_rsum)
    if best_score:
        log_str3 += ' (best %s: %.2f)' %(args.val_metric, best_score)

    i2t_recall, t2i_recall = (r1, r5, r10), (r1i, r5i, r10i) 
    
    logging.info(log_str1)
    logging.info(log_str2)
    logging.info(log_str3)

    dscr = get_description(args, epoch)
    log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)
    lock_and_write_to_file(args.log_file, log_msg)

    if args.val_metric == 'rsum':
        return rsum, i2t_recall, t2i_recall, recall_1k
    elif args.val_metric == 'med_rsum':
        return med_rsum, i2t_recall, t2i_recall, recall_1k
    else:
        return mean_rsum, i2t_recall, t2i_recall, recall_1k


def update_best_score(new_score, old_score, is_higher_better):
    if not old_score:
        score, updated = new_score, True
    else:
        if is_higher_better:
            score = max(new_score, old_score)
            updated = new_score > old_score
        else:
            score = min(new_score, old_score)
            updated = new_score < old_score
    return score, updated

def warmup(model, epoch, args, multi_gpu):
    if args.img_finetune and args.txt_finetune:
        warm = epoch >= args.warm_epoch
        if args.warm_img:
            for idx, param in enumerate((model.module if multi_gpu else model).img_enc.cnn.parameters()):
                param.requires_grad = warm
        if args.warm_txt:
            (model.module if multi_gpu else model).txt_enc.embed.weight.requires_grad = warm

def finetune_lr_lower(optimizer, epoch, args):
    if epoch == args.warm_epoch:
        for g in optimizer.param_groups:
            g['lr'] *= args.finetune_lr_lower

def tri_mean_to_max(criterion, epoch, args):
    if args.tri_mean_to_max:
        criterion.max_violation = epoch >= args.warm_epoch      

def main():
    args = verify_input_args(parser.parse_args())
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        args = copy_input_args_from_ckpt(args, ckpt['args'])
    print(args)
    
    LOG_DIR = os.path.join(args.log_dir, args.remark)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    args.log_dir = LOG_DIR
    args.logger_name = LOG_DIR
    wandb.init(project='cross_modal_retrieval', notes=args.log_dir, name = args.remark)
    wandb.config.update(args)

    # Load Vocabulary Wrapper
    vocab_path = os.path.join(args.vocab_path, '%s_vocab.pkl' % args.data_name)
    vocab = pickle.load(open(vocab_path, 'rb'))
    vocab.add_word('<mask>')
    print('Add <mask> token into the vocab')

    # Dataloaders
    if args.fast_batch:
        if args.data_name in ['coco', 'f30k', 'coco_butd', 'f30k_butd']:
            txt_per_img = 5 
        elif 'cub' in args.data_name:
            txt_per_img = 10
        else:
            raise NotImplementedError
    else:
        txt_per_img = 1

    if args.data_name in ['coco', 'f30k', 'coco_butd', 'f30k_butd']:
        trn_loader, val_loader = data.get_loaders(args, vocab)
        test_dataset, test_loader = None, data.get_test_loader(args, vocab)
    # elif args.data_name in ['cub_trainval1', 'cub_trainval2', 'cub_trainval3', 'cub']:
    #     dls = data_cub.prepare_cub_dataloaders(args, vocab, args.data_name, './data/CUB_200_2011_caption', './data/CUB_200_2011_caption/text_c10')
    #     _, trn_loader = dls['train']
    #     test_dataset, test_loader = dls['val']
    else:
        raise NotImplementedError

    # Construct the model
    if 'butd' in args.data_name:
        model = PVSE_perceiver(vocab.word2idx, args)
    else:
        if args.arch == 'pvse' or args.arch == 'fc':
            model = PVSE(vocab.word2idx, args)
        elif args.arch == 'perceiver': 
            model = PVSE_perceiver(vocab.word2idx, args)
        else:
            raise NotImplementedError
            
    if torch.cuda.is_available():
        if args.multi_gpu:
            model = nn.DataParallel(model)
        if args.sync_bn:
            model = convert_model(model)
        model = model.cuda()
        cudnn.benchmark = True
        
    wandb.watch(models=model, log_freq=1000, log='gradients')

    # optionally resume from a ckpt
    if args.ckpt:
        target_vocab_path = './vocab/%s_vocab.pkl' % args.data_name
        src_vocab_path = './vocab/%s_vocab.pkl' % ckpt['args'].data_name
        if target_vocab_path != src_vocab_path:
            print('Vocab mismatch!')
            sys.exit(-1)
        model.load_state_dict(ckpt['model'])
        
    # distance function options
    train_distance = SetwiseDistance(args.img_num_embeds, args.txt_num_embeds*5 if args.txts_as_set else args.txt_num_embeds,\
        args.denominator, args.temperature, args.temperature_txt_scale, args.alpha)
    if args.loss == 'smooth_chamfer':
        train_distance_fn = train_distance.smooth_chamfer_distance
    elif args.loss == 'smooth_chamfer_self':   
        train_distance_fn = train_distance.smooth_chamfer_self
    elif args.loss == 'smooth_chamfer_t2i':
        train_distance_fn = train_distance.smooth_chamfer_distance_t2i
    elif args.loss == 'chamfer':
        train_distance_fn = train_distance.chamfer_distance
    elif args.loss == 'max':
        train_distance_fn = train_distance.max_distance
    elif args.loss == 'dropped_sc':
        train_distance_fn = train_distance.smooth_chamfer_distance_dropped
    elif args.loss == 'mp':
        train_distance_fn = train_distance.avg_distance
    elif 'old' in args.loss:
        train_distance_fn = None
    else:
        assert False
    
    eval_distance = SetwiseDistance(args.img_num_embeds, args.txt_num_embeds, \
        args.denominator, args.temperature, args.temperature_txt_scale)
    if args.eval_distance == 'smooth_chamfer':
        eval_distance_fn = eval_distance.smooth_chamfer_distance
    elif args.eval_distance == 'smooth_chamfer_t2i':
        eval_distance_fn = eval_distance.smooth_chamfer_distance_t2i
    elif args.eval_distance == 'smooth_chamfer_self':
        eval_distance_fn = eval_distance.smooth_chamfer_self
    elif args.eval_distance == 'chamfer':
        eval_distance_fn = eval_distance.chamfer_distance
    elif args.eval_distance == 'max':
        eval_distance_fn = eval_distance.max_distance
    elif args.eval_distance == 'dropped_sc':
        eval_distance_fn = eval_distance.smooth_chamfer_distance_dropped
    elif args.loss == 'mp':
        eval_distance_fn = train_distance_fn
    else:
        assert False
            
    # Loss and optimizer
    if args.loss == 'pvse_old':
        criterion = PVSELoss(args)
    elif args.loss == 'chamfer_old':
        criterion = ChamferLoss(args, smooth=False)
    elif args.loss == 'smooth_chamfer_old':
        criterion = ChamferLoss(args, smooth=True)
    elif args.loss in ['smooth_chamfer', 'smooth_chamfer_self', 'chamfer', 'max', 'smooth_chamfer_t2i', 'dropped_sc', 'mp']:
        # assert args.fast_batch
        criterion = AsymmetricTripletLoss(
            img_set_size=args.img_num_embeds, 
            txt_set_size=args.txt_num_embeds*5 if args.txts_as_set else args.txt_num_embeds, 
            distance_fn=train_distance_fn, 
            opt=args, txt_per_img=txt_per_img if args.fast_batch else 1, is_cosine=True
        )
    else:
        assert False
        
    module = model.module if args.multi_gpu else model
    param_groups = [
        {'params': list(set(module.img_enc.parameters()).difference(set(module.img_enc.pie_net.parameters()))), 'lr': args.lr},
        {'params': module.img_enc.pie_net.parameters(), 'lr': args.lr * args.img_pie_lr_scale},
        {'params': list(set(module.txt_enc.parameters()).difference(set(module.txt_enc.pie_net.parameters()))), 'lr': args.lr * args.txt_lr_scale},
        {'params': module.txt_enc.pie_net.parameters(), 'lr': args.lr * args.txt_pie_lr_scale}
    ]
    
    if args.loss == 'mp':
        param_groups += [{'params':train_distance.parameters(), 'lr': args.lr}]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        from adamp import AdamP
        optimizer = AdamP(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-10, verbose=True)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trn_loader)*args.num_epochs)
    elif args.lr_scheduler == 'multi_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma = args.lr_step_gamma)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma = args.lr_step_gamma)
    elif args.lr_scheduler == 'pvse_cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch, T_mult=1)
        args.finetune_lr_lower = 0.1
        
    # Train resume setting
    if args.ckpt and 'best_score' in ckpt and ckpt['args'].val_metric == args.val_metric:
        best_score = ckpt['best_score']
    else:
        best_score = None
    
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print("###Training###")
    for epoch in range(args.num_epochs):
        #warm up training data
        warmup(model, epoch, args, args.multi_gpu)
        finetune_lr_lower(optimizer, epoch, args)
        tri_mean_to_max(criterion, epoch, args)

        if args.lr_scheduler == 'pvse_cosine' and epoch == args.warm_epoch:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch, T_mult=1)
        
        # train for one epoch
        warm_iter = len(trn_loader) * args.lr_warmup
        loss, losses_dict, stat_dict = train(
            epoch, trn_loader, model, criterion, optimizer, scaler, args, 
            lr_warmup=warm_iter, scheduler=lr_scheduler if args.lr_scheduler == 'cosine' else None)
        
        wandb.log({"epoch": epoch}, step=total_iter)
        wandb.log({"Loss": loss}, step=total_iter)
        for key, val in losses_dict.items():
            wandb.log({key: val.avg}, step=total_iter)
        for key, val in stat_dict.items():
            wandb.log({key: val.avg}, step=total_iter)
        wandb.log({"LR" : optimizer.param_groups[0]['lr']}, step=total_iter)
        
        print("###validate###")
        # evaluate on validation set
        with torch.no_grad():
            if epoch % args.eval_epoch == 0:
                val_score, i2t_recall, t2i_recall, recall_1k = validate(None, val_loader, model, args, eval_distance_fn, True, epoch, best_score)
                wandb.log({"val i2t R@1" : i2t_recall[0]}, step=total_iter)
                wandb.log({"val i2t R@5" : i2t_recall[1]}, step=total_iter)
                wandb.log({"val i2t R@10" : i2t_recall[2]}, step=total_iter)

                wandb.log({"val t2i R@1" : t2i_recall[0]}, step=total_iter)
                wandb.log({"val t2i R@5" : t2i_recall[1]}, step=total_iter)
                wandb.log({"val t2i R@10" : t2i_recall[2]}, step=total_iter)
                
                rsum_val = i2t_recall[0]+i2t_recall[1]+i2t_recall[2]+t2i_recall[0]+t2i_recall[1]+t2i_recall[2]
                wandb.log({"val rsum": rsum_val}, step=total_iter)

                val_score, i2t_recall, t2i_recall, recall_1k = validate(test_dataset, test_loader, model, args, eval_distance_fn, False, epoch, best_score)
                wandb.log({"i2t R@1" : i2t_recall[0]}, step=total_iter)
                wandb.log({"i2t R@5" : i2t_recall[1]}, step=total_iter)
                wandb.log({"i2t R@10" : i2t_recall[2]}, step=total_iter)

                wandb.log({"t2i R@1" : t2i_recall[0]}, step=total_iter)
                wandb.log({"t2i R@5" : t2i_recall[1]}, step=total_iter)
                wandb.log({"t2i R@10" : t2i_recall[2]}, step=total_iter)
                
                wandb.log({"1k i2t R@1" : recall_1k[0]}, step=total_iter)
                wandb.log({"1k i2t R@5" : recall_1k[1]}, step=total_iter)
                wandb.log({"1k i2t R@10" : recall_1k[2]}, step=total_iter)
                wandb.log({"1k t2i R@1" : recall_1k[3]}, step=total_iter)
                wandb.log({"1k t2i R@5" : recall_1k[4]}, step=total_iter)
                wandb.log({"1k t2i R@10" : recall_1k[5]}, step=total_iter)
                
                rsum_1k = recall_1k[0]+recall_1k[1]+recall_1k[2]+recall_1k[3]+recall_1k[4]+recall_1k[5]
                rsum_5k = i2t_recall[0]+i2t_recall[1]+i2t_recall[2]+t2i_recall[0]+t2i_recall[1]+t2i_recall[2]
                
                wandb.log({"1k rsum": rsum_1k}, step=total_iter)
                wandb.log({"5k rsum": rsum_5k}, step=total_iter)
                
                # remember best rsum and save ckpt
                best_score, updated = update_best_score(rsum_1k, best_score, args.val_metric=='rsum')
                save_ckpt({
                    'args': args,
                    'epoch': epoch,
                    'best_score': best_score,
                    'model': model.state_dict(),
                }, updated, prefix=args.logger_name + '/')
                
                if epoch == 49:
                    torch.save({
                        'args': args,
                        'epoch': epoch,
                        'best_score': best_score,
                        'model': model.state_dict(),
                    }, 'epoch_49.pth.tar')
        
        # adjust learning rate if rsum stagnates
        if epoch < args.lr_scheduling_stop_epoch:
            if args.lr_scheduler == 'plateau' and epoch >= args.lr_warmup_epoch:
                lr_scheduler.step(val_score)
            elif args.lr_scheduler != 'cosine':
                lr_scheduler.step()

if __name__ == '__main__':
    main()
