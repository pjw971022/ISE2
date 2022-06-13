import os, sys, pdb

import argparse
parser = argparse.ArgumentParser(description='Parameters for training PVSE')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Names, paths, logging, etc
parser.add_argument('--data_name', default='coco', choices=('cub','coco', 'f30k', 'coco_butd', 'f30k_butd'), help='Dataset name (coco|cub)')
parser.add_argument('--data_path', default=CUR_DIR+'/data/', help='path to datasets')
parser.add_argument('--vocab_path', default=CUR_DIR+'/vocab/', help='Path to saved vocabulary pickle files')
parser.add_argument('--logger_name', default=CUR_DIR+'/runs/runX', help='Path to save the model and logs')
parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log')
parser.add_argument('--log_file', default=CUR_DIR+'/logs/logX.log', help='Path to save result logs') 
parser.add_argument('--debug', action='store_true', help='Debug mode: use 1/10th of training data for fast iteration')
parser.add_argument('--log_dir', default=CUR_DIR+'/logs/', help='Path to save result logs') 
parser.add_argument('--sample_dir', default=CUR_DIR+'/attn_samples/', help='Path to save attn map samples') 

# Data parameters
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding')
parser.add_argument('--workers', default=16, type=int, help='Number of data loader workers')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input')
parser.add_argument('--max_video_length', default=1, type=int, help='Maximum length of video sequence')

# Model parameters
parser.add_argument('--cnn_type', default='resnet152', help='The CNN used for image encoder')
parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding')
parser.add_argument('--wemb_type', default='none', choices=('glove','fasttext','none'), type=str, help='Word embedding (glove|fasttext)')
parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss')
parser.add_argument('--order', action='store_true', help='Enable order embedding')

# Attention parameters
parser.add_argument('--img_attention', action='store_true', help='Use self attention on images/videos')
parser.add_argument('--txt_attention', action='store_true', help='Use self attention on text')

parser.add_argument('--img_num_embeds', default=1, type=int, help='Number of embeddings for MIL formulation')
parser.add_argument('--txt_num_embeds', default=1, type=int, help='Number of embeddings for MIL formulation')

# Loss weights
parser.add_argument('--mmd_weight', default=.0, type=float, help='Weight term for the MMD loss')
parser.add_argument('--div_weight', default=.0, type=float, help='Weight term for the log-determinant divergence loss')
parser.add_argument('--kl_weight', default=0.00001, type=float, help='Weight term for the kl loss')
parser.add_argument('--unif_weight', default=.0, type=float, help='Weight term for the uniformity loss')
parser.add_argument('--qreg_weight', default=.0, type=float, help='Weight term for the uniformity loss')
parser.add_argument('--a', default=15., type=float, help='Weight term for the kl loss')
parser.add_argument('--b', default=15., type=float, help='Weight term for the uniformity loss')
parser.add_argument('--sigma', default=1., type=float, help='inverse temperature value used when computing scd')
parser.add_argument('--loss', type=str)

# Training / optimizer setting
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image embedding')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embedding')
parser.add_argument('--val_metric', default='rsum', choices=('rsum','med_rsum','mean_rsum'), help='Validation metric to use (rsum|med_rsum|mean_rsum)')
parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch')
parser.add_argument('--batch_size_eval', default=256, type=int, help='Size of a evaluation mini-batch')
parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay (l2 norm) for optimizer')
parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate')
parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='path to latest ckpt (default: none)')
parser.add_argument('--ckpt2', default='', type=str, metavar='PATH', help='path to latest ckpt (default: none)')
parser.add_argument('--eval_on_gpu', action='store_true', help='Evaluate on GPU (default: CPU)')
parser.add_argument('--legacy', action='store_true', help='Turn this on to reproduce results in CVPR2018 paper')

# customized settings
parser.add_argument('--warm_epoch', default=30, type=int, help='warm up epochs')
parser.add_argument('--remark', type=str)
parser.add_argument('--lr_scheduler', type=str, default='cosine')
parser.add_argument('--lr_milestones', nargs='+', type=int ,help='step value used in step scheduler')
parser.add_argument('--lr_step_gamma', type=float, default=0.5, help='step value used in step scheduler')
parser.add_argument('--lr_step_size', type=int, help='step value used in step scheduler')
parser.add_argument('--warm_txt', action='store_true')
parser.add_argument('--warm_img', action='store_true')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--sync_bn', action='store_true')
parser.add_argument('--fast_batch', action='store_true')
parser.add_argument('--fast_1x1', action='store_true')
parser.add_argument('--i2t_weight', type=float, default=1.0) # for debug 
parser.add_argument('--t2i_weight', type=float, default=1.0) # for debug
parser.add_argument('--semi_hard_triplet', action='store_true')
parser.add_argument('--img_pie_lr_scale', type=float, default=1.0)
parser.add_argument('--txt_pie_lr_scale', type=float, default=1.0)
parser.add_argument('--txt_lr_scale', type=float, default=1.0)
parser.add_argument('--denominator', type=float, default=2.0, help='Denominator of chamfer/smooth chamfer distance, it acts as a average between two sets when = 2')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature of chamfer/smooth chamfer distance')
parser.add_argument('--temperature_txt_scale', type=float, default=1.0, help='Temperature of chamfer/smooth chamfer distance')
parser.add_argument('--alpha', type=float, default=0.1, help='smooth chamfer distance Pnorm scaling')
parser.add_argument('--eval_distance', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--finetune_lr_lower', type=float, default=1.0)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--eval_mode', type=int,help='mode 0: evalrank mode 1: visualize')

# GPO options
parser.add_argument('--lr_warmup', type=int, default=0)
parser.add_argument('--bn_eval', action='store_true')
parser.add_argument('--tri_mean_to_max', action='store_true')
parser.add_argument('--gpo_rnn', action='store_true')
parser.add_argument('--gpo_1x1', action='store_true')

# perceiver options
parser.add_argument('--perceiver_pre_self_attns', type=int, default=1)
parser.add_argument('--perceiver_query_self_attns', type=int, default=0)
parser.add_argument('--perceiver_self_per_cross_attn', type=int, default=0)
parser.add_argument('--perceiver_depth', type=int, default=1)
parser.add_argument('--perceiver_cross_head', type=int, default=4)
parser.add_argument('--perceiver_cross_dim', type=int, default=64)

parser.add_argument('--perceiver_img_latent_head', type=int, default=8)
parser.add_argument('--perceiver_img_latent_dim', type=int, default=32)
parser.add_argument('--perceiver_txt_latent_head', type=int, default=8)
parser.add_argument('--perceiver_txt_latent_dim', type=int, default=32)

parser.add_argument('--perceiver_img_pos_enc_type', type=str, default='none')
parser.add_argument('--perceiver_txt_pos_enc_type', type=str, default='none')

parser.add_argument('--perceiver_last_fc', action='store_true')
parser.add_argument('--perceiver_input_dim', type=int, default=1024)
parser.add_argument('--perceiver_query_dim', type=int, default=1024)
parser.add_argument('--perceiver_1x1', action='store_true')
parser.add_argument('--perceiver_pre_norm', action='store_true')
parser.add_argument('--perceiver_post_norm', action='store_true')
parser.add_argument('--perceiver_activation', type=str, default='geglu')
parser.add_argument('--perceiver_last_ln', action='store_true')
parser.add_argument('--perceiver_residual', action='store_true')
parser.add_argument('--perceiver_residual_norm', action='store_true')
parser.add_argument('--perceiver_weight_sharing', action='store_true')
parser.add_argument('--perceiver_ff_mult', type=int, default=2)
parser.add_argument('--perceiver_residual_activation', type=str, default='none')
parser.add_argument('--perceiver_residual_fc', action='store_true')
parser.add_argument('--perceiver_xavier_init', action='store_true')
parser.add_argument('--perceiver_more_dropout', action='store_true')
parser.add_argument('--perceiver_thin_ff', action='store_true')
parser.add_argument('--rnn_no_dropout', action='store_true')

# Options for query and attention mechanisms
parser.add_argument('--perceiver_cross_attn_type', default='transformer', choices=('slot', 'transformer'))
parser.add_argument('--query_xavier_init', action='store_true')
parser.add_argument('--query_fixed', action='store_true')
parser.add_argument('--query_slot', action='store_true')

parser.add_argument('--lr_scheduling_stop_epoch', type=int, default=50000)
parser.add_argument('--txts_as_set', action='store_true')

parser.add_argument('--txt_attention_input', type=str)
parser.add_argument('--txt_attention_head', type=str)
parser.add_argument('--txt_pooling', type=str)

parser.add_argument('--img_res_first_fc', action='store_true')
parser.add_argument('--img_res_last_fc', action='store_true')
parser.add_argument('--img_res_first_pool', action='store_true')
parser.add_argument('--img_res_pool', default='avg', choices=('avg','max', 'avgmax', 'minmax'))
parser.add_argument('--enc_res', action='store_true')
parser.add_argument('--learned_at_attn', action='store_true', help='The way how query treated in DETR')
parser.add_argument('--res_gru', action='store_true')
parser.add_argument('--detach_global', action='store_true')
parser.add_argument('--res_act_local_dropout', type=float, default=0)
parser.add_argument('--res_act_global_dropout', type=float, default=0)
parser.add_argument('--img_1x1_dropout', type=float, default=0)

parser.add_argument('--unif_residual', action='store_true', help='Flag for whether to use residual features in computing uniformity loss')

# CUB dataloader options
parser.add_argument('--random_erasing_prob', type=float, default=0)
parser.add_argument('--caption_drop_prob', type=float, default=0)
parser.add_argument('--butd_drop_prob', type=float, default=0)
parser.add_argument('--drop_last', action='store_true')
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--cub_sampler', default='pcme', choices=('shuffle','pcme', 'balanced'))

parser.add_argument('--ten_crop',action='store_true')
parser.add_argument('--ten_crop_idx',type=int)
parser.add_argument('--only_res',action='store_true')

parser.add_argument('--use_bert', action='store_true')

def verify_input_args(args):
  # Process input arguments
  if args.ckpt:
    # assert os.path.isfile(args.ckpt), 'File not found: {}'.format(args.ckpt)
    pass
  if (args.img_num_embeds > 1 or args.txt_num_embeds > 1) and (args.img_attention and args.txt_attention) is False:
    print('When num_embeds > 1, both img_attention and txt_attention must be True')
    sys.exit(-1)
  if not (args.val_metric=='rsum' or args.val_metric=='med_rsum' or args.val_metric=='mean_rsum'):
    print('Unknown validation metric {}'.format(args.val_metric))
    sys.exit(-1)
  if not os.path.isdir(args.logger_name):
    print('Creading a directory: {}'.format(args.logger_name))
    os.makedirs(args.logger_name)
  if args.log_file and not os.path.isdir(os.path.dirname(args.log_file)):
    print('Creating a directory: {}'.format(os.path.dirname(args.log_file)))
    os.makedirs(os.path.dirname(args.log_file))
  return args
