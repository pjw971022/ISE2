from __future__ import print_function
import os, sys
import pickle
import time
import glob

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model import PVSE
from model_perceiver import PVSE_perceiver
from loss import cosine_sim, order_sim
from vocab import Vocabulary
from data import get_test_loader, get_loaders
# from data_cub import prepare_cub_dataloaders
# from eval_cub import i2t_cub, t2i_cub
from logger import AverageMeter
from option import parser, verify_input_args
from distance import SetwiseDistance
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
ORDER_BATCH_SIZE = 100
def image_saving(im_batch, att_mat_batch, sample_dir,ids):
    for i, im in enumerate(im_batch):
        SET  = att_mat_batch[i] 
        for element in SET:
            att_mat = element
            print('att_mat.shape: ',att_mat.shape)
            # att_mat = torch.stack(att_mat).squeeze(1)

            # Average the attention weights across all heads.
            # att_mat = torch.mean(att_mat, dim=1)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(0)).cuda()
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            print(f'aug_att_mat: {aug_att_mat.shape}')
            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]
            print(f'joint_attentions: {joint_attentions.shape}')    
            # Attention from the output token to the input space.
            v = joint_attentions[-1].unsqueeze(dim=1)
            print(f'v: {v.shape}')    
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v.reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
            result = (mask * im).astype("uint8")

            img_list = [im, result]
            image_cat = torch.cat(img_list, dim=3)
            
            sample_path = os.path.join(sample_dir, '{}-images.jpg'.format(ids))
            save_image(image_cat, sample_path)
        break
    #save_image( [이미지 텐서 변수], [저장 경로])

    
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    # ax1.set_title('Original')
    # ax2.set_title('Attention Map')
    # _ = ax1.imshow(im)
    # _ = ax2.imshow(result)



def visualize(model, args, split='test'):
    print('Loading dataset')
    if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd']:
        if split == 'val':
            _, data_loader = get_loaders(args, vocab)
            dataset = data_loader.dataset
        elif split == 'test':
            dataset, data_loader = None, get_test_loader(args, vocab)

    # switch to evaluate mode
    model.eval()
    # numpy array to keep all the embeddings
    butd = 'butd' in args.data_name
    print('\n###### Visualizing ######')
    img_embs, txt_embs = None, None
    for i, data in tqdm(enumerate(data_loader)):
        if butd:
            img, txt, img_len, txt_len, ids = data
            img, txt, img_len, txt_len = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda()
            # print(f"img.shape:{img.shape}, txt.shape:{txt.shape}, dataset_ids:{ids}")
        
        else:
            img_len = None
            img, txt, txt_len, ids = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        # compute the embeddings
        img_emb, txt_emb, img_attn, txt_attn, _, _ = model.forward(img, txt, img_len, txt_len)
        print(f'img_attn_shape: {img_attn.shape}, img_shape: {img.shape}')
        image_saving(img, img_attn, args.sample_dir,ids)
        del img, txt, img_len, txt_len
        break
    

def encode_data(model, data_loader, butd, use_gpu=False):
    """Encode all images and sentences loadable by data_loader"""
    # switch to evaluate mode
    model.eval()

    use_mil = model.module.mil if hasattr(model, 'module') else model.mil

    # numpy array to keep all the embeddings
    img_embs, txt_embs = None, None
    for i, data in tqdm(enumerate(data_loader)):
        if butd:
            img, txt, img_len, txt_len, ids = data
            img, txt, img_len, txt_len = img.cuda(), txt.cuda(), img_len.cuda(), txt_len.cuda()
            # print(f"img.shape:{img.shape}, txt.shape:{txt.shape}, dataset_ids:{ids}")
        
        else:
            img_len = None
            img, txt, txt_len, ids = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        # compute the embeddings
        img_emb, txt_emb, img_attn, txt_attn, _, _ = model.forward(img, txt, img_len, txt_len)
        del img, txt, img_len, txt_len

        # initialize the output embeddings
        if img_embs is None:
            img_emb_sz = [len(data_loader.dataset), img_emb.size(1), img_emb.size(2)] \
                    if use_mil else [len(data_loader.dataset), img_emb.size(1)]
            txt_emb_sz = [len(data_loader.dataset), txt_emb.size(1), txt_emb.size(2)] \
                    if use_mil else [len(data_loader.dataset), txt_emb.size(1)]
            img_embs = torch.zeros(img_emb_sz, dtype=img_emb.dtype, requires_grad=False).cuda()
            txt_embs = torch.zeros(txt_emb_sz, dtype=txt_emb.dtype, requires_grad=False).cuda()

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb 
        txt_embs[ids] = txt_emb 
        
    return img_embs, txt_embs


def i2t(images, sentences, distance_fn, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False, is_cosine=True):
    """
    Images->Text (Image Annotation)
    Images: (nreps*N, K) matrix of images
    Captions: (nreps*N, K) matrix of sentences
    """
    if use_gpu:
        assert not order, 'Order embedding not supported in GPU mode'

    # NOTE nreps : numbrt of captions per image, npts: number of images
    if npts is None:
        npts = int(images.shape[0] / nreps)
        
    index_list = []
    ranks, top1 = np.zeros(npts), np.zeros(npts)
    for index in range(npts):
        # Get query image
        im = images[nreps * index]
        im = im.reshape((1,) + im.shape)
        # Compute scores
        if use_gpu:
            if len(sentences.shape) == 2:
                sim = im.mm(sentences.t()).view(-1)
            else:
                _, K, D = im.shape
                sim = distance_fn(is_cosine, im.view(-1, D), sentences.view(-1, D)).flatten()
        else: 
            if order:
                if index % ORDER_BATCH_SIZE == 0:
                    mx = min(images.shape[0], nreps * (index + ORDER_BATCH_SIZE))
                    im2 = images[nreps * index:mx:nreps]
                    sim_batch = order_sim(torch.Tensor(im2).cuda(), torch.Tensor(sentences).cuda())
                    sim_batch = sim_batch.cpu().numpy()
                sim = sim_batch[index % ORDER_BATCH_SIZE]
            else:
                sim = np.tensordot(im, sentences, axes=[2, 2]).max(axis=(0,1,3)).flatten() \
                        if len(sentences.shape) == 3 else np.dot(im, sentences.T).flatten()

        if use_gpu:
            _, inds_gpu = sim.sort()
            inds = inds_gpu.cpu().numpy().copy()[::-1] #reverse order / change it to descending order
        else:
            inds = np.argsort(sim)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(nreps * index, nreps * (index + 1), 1):
            tmp = np.where(inds == i)[0][0] # find the rank of given text data
            if tmp < rank:
                rank = tmp
            # find highest rank among matching queries
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    # import ipdb; ipdb.set_trace()
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, sentences, distance_fn, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False, is_cosine=True):
    """
    Text->Images (Image Search)
    Images: (nreps*N, K) matrix of images
    Captions: (nreps*N, K) matrix of sentences
    """
    if use_gpu:
        assert not order, 'Order embedding not supported in GPU mode'

    if npts is None:
        npts = int(images.shape[0] / nreps)

    if use_gpu:
        ims = torch.stack([images[i] for i in range(0, len(images), nreps)])
    else:
        ims = np.array([images[i] for i in range(0, len(images), nreps)])

    ranks, top1 = np.zeros(nreps * npts), np.zeros(nreps * npts)
    for index in range(npts):
        # Get query sentences
        queries = sentences[nreps * index:nreps * (index + 1)]

        # Compute scores
        if use_gpu:
            if len(sentences.shape) == 2:
                sim = queries.mm(ims.t())
            else:
                sim = distance_fn(is_cosine, ims.view(-1, ims.size(-1)), queries.view(-1, queries.size(-1))).t()
        else:
            if order:
                if nreps * index % ORDER_BATCH_SIZE == 0:
                    mx = min(sentences.shape[0], nreps * index + ORDER_BATCH_SIZE)
                    sentences_batch = sentences[nreps * index:mx]
                    sim_batch = order_sim(torch.Tensor(images).cuda(), 
                                                                torch.Tensor(sentences_batch).cuda())
                    sim_batch = sim_batch.cpu().numpy()
                sim = sim_batch[:, (nreps * index) % ORDER_BATCH_SIZE:(nreps * index) % ORDER_BATCH_SIZE + nreps].T
            else:
                sim = np.tensordot(queries, ims, axes=[2, 2]).max(axis=(1,3)) \
                        if len(sentences.shape) == 3 else np.dot(queries, ims.T)

        inds = np.zeros(sim.shape)
        for i in range(len(inds)):
            if use_gpu:
                _, inds_gpu = sim[i].sort()
                inds[i] = inds_gpu.cpu().numpy().copy()[::-1]
            else:
                inds[i] = np.argsort(sim[i])[::-1]
            ranks[nreps * index + i] = np.where(inds[i] == index)[0][0]
            top1[nreps * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)



def convert_old_state_dict(x, model, multi_gpu=False):
    params = model.state_dict()
    prefix = ['module.img_enc.', 'module.txt_enc.'] \
            if multi_gpu else ['img_enc.', 'txt_enc.']
    for i, old_params in enumerate(x):
        for key, val in old_params.items():
            key = prefix[i] + key.replace('module.','').replace('our_model', 'pie_net')
            assert key in params, '{} not found in model state_dict'.format(key)
            params[key] = val
    return params



def evalrank(model, args, split='test'):
    print('Loading dataset')
    if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd']:
        if split == 'val':
            _, data_loader = get_loaders(args, vocab)
            dataset = data_loader.dataset
        elif split == 'test':
            dataset, data_loader = None, get_test_loader(args, vocab)
    # elif args.data_name in ['cub_trainval1', 'cub_trainval2', 'cub_trainval3', 'cub']:
    #     dls = prepare_cub_dataloaders(args, vocab, args.data_name, './data/CUB_200_2011_caption', './data/CUB_200_2011_caption/text_c10')
    #     dataset, data_loader = dls['val']

    print('Computing results... (eval_on_gpu={})'.format(args.eval_on_gpu))
    img_embs, txt_embs = encode_data(model, data_loader, 'butd' in args.data_name, args.eval_on_gpu)
    n_samples = img_embs.shape[0]

    nreps = 5 if args.data_name in ['f30k', 'coco', 'coco_butd', 'f30k_butd'] else 10
    print('Images: %d, Sentences: %d' % (img_embs.shape[0] / nreps, txt_embs.shape[0]))
    
    img_set_size, txt_set_size = args.img_num_embeds, args.txt_num_embeds
    distance = SetwiseDistance(img_set_size, txt_set_size, args.denominator, args.temperature, args.temperature_txt_scale)
    if args.loss == 'smooth_chamfer':
        distance_fn = distance.smooth_chamfer_distance
    elif args.loss == 'chamfer':
        distance_fn = distance.chamfer_distance
    elif args.loss == 'max':
        distance_fn = distance.max_distance
    else:
        raise NotImplementedError
    
    mean_metrics = None
    if 'coco' in args.data_name and split == 'test':
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
            
            r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
            print("Image to text: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % r)
            ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
            print("Text to image: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % ri)

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

        print("-----------------------------------")
        print("Mean metrics from 5-fold evaluation: ")
        print("rsum: %.2f" % (mean_metrics[-1]))
        print("Average i2t Recall: %.2f" % mean_metrics[-3])
        print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[:7])
        print("Average t2i Recall: %.2f" % mean_metrics[-2])
        print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[7:14])

    # no cross-validation, full evaluation
    # if 'cub' in args.data_name:
    #     r, rt = i2t_cub(
    #         img_embs, txt_embs, 
    #         dataset.index_to_class, dataset.class_to_indices,
    #         distance_fn, nreps=nreps, return_ranks=True)
    #     ri, rti = t2i_cub(
    #         img_embs, txt_embs,
    #         dataset.index_to_class, dataset.class_to_indices, 
    #         distance_fn, nreps=nreps, return_ranks=True)
    # else:
    # r, rt = i2t(img_embs, txt_embs, distance_fn, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
    # ri, rti = t2i(img_embs, txt_embs, distance_fn, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
        
    # ar = (r[0] + r[1] + r[2]) / 3
    # ari = (ri[0] + ri[1] + ri[2]) / 3
    # rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    # r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
    # ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
    # print("rsum: %.2f" % rsum)
    # print("Average i2t Recall: %.2f" % ar)
    # print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % r)
    # print("Average t2i Recall: %.2f" % ari)
    # print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % ri)
    
    return mean_metrics


if __name__ == '__main__':
    args = verify_input_args(parser.parse_args())
    opt = verify_input_args(parser.parse_args())

    # load vocabulary used by the model
    with open('./vocab/%s_vocab.pkl' % args.data_name, 'rb') as f:
        vocab = pickle.load(f)
    args.vocab_size = len(vocab)
    vocab.add_word('<mask>')
    print(args.vocab_size)
    # load model and options
    assert os.path.isfile(args.ckpt)
    
    if args.arch == 'pvse':
        model = PVSE(vocab.word2idx, args)
    elif args.arch == 'perceiver':
        print(len(vocab.word2idx))
        model = PVSE_perceiver(vocab.word2idx, args)
        
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda() if args.multi_gpu else model.cuda()
        torch.backends.cudnn.benchmark = True
    
    # Reproduced weight
    state_dict = torch.load(args.ckpt)['model']
    print(torch.load(args.ckpt)['best_score'])
    # convert_old_state_dict(state_dict,model)
    # model.load_state_dict(convert_old_state_dict(state_dict,model))
    model.load_state_dict(state_dict)
    if args.eval_mode==0:
        with torch.no_grad():
            # evaluate
            # metrics = evalrank(model, args, split='val')
            metrics = evalrank(model, args, split='test')
    elif args.eval_mode==1:
        with torch.no_grad():
            visualize(model, args, split='test')