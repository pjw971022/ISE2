#!/bin/sh
#SBATCH -J ISE_train_resnet
#SBATCH -o ISE_train_resnet.out
#SBATCH -e ISE_train_resnet.err
#SBATCH --time 60:00:00
#SBATCH --gres=gpu:1



#  example train script for coco (ResNet)
python3 train.py \
--data_name coco --cnn_type resnet152 --wemb_type glove \
--margin 0.1 --max_violation --img_num_embeds 2 --txt_num_embeds 2 \
--img_attention --txt_attention --img_finetune --txt_finetune \
--mmd_weight 0.01 --div_weight 0.0 --unif_weight 0.1 \
--batch_size 200 --warm_epoch 10 --num_epochs 120 \
--optimizer adam --lr_scheduler step --lr_step_size 10 --lr_step_gamma 0.5 \
--warm_img --finetune_lr_lower 1 --log_step 200 \
--lr 2e-3 --txt_lr_scale 1 --img_pie_lr_scale 1 --txt_pie_lr_scale 1 \
--eval_on_gpu --sync_bn --amp --fast_batch \
--loss smooth_chamfer --eval_distance smooth_chamfer --temperature 16 --txt_pooling rnn \
--arch perceiver --txt_attention_head transformer --txt_attention_input wemb \
--perceiver_img_pos_enc_type sine --perceiver_txt_pos_enc_type sine \
--perceiver_1x1 --perceiver_residual --perceiver_residual_norm --perceiver_residual_activation sigmoid \
--perceiver_activation relu \
--perceiver_pre_self_attns 1 --perceiver_query_self_attns 0 --perceiver_self_per_cross_attn 1 \
--perceiver_img_latent_head 8 --perceiver_img_latent_dim 32 \
--perceiver_txt_latent_head 8 --perceiver_txt_latent_dim 32 \
--perceiver_ff_mult 2 \
--perceiver_cross_head 4 --perceiver_cross_dim 64 \
--img_res_pool avg --img_res_last_fc \
--perceiver_input_dim 256 --perceiver_query_dim 1024 --embed_size 1024 --perceiver_pre_norm \
--remark coco_resnet_i2t2_1024 \
--workers 4 --grad_clip 1 --dropout 0.1
