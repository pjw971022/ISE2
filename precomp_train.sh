#!/bin/sh
#SBATCH -J ISE_train_precomp
#SBATCH -o ISE_train_precomp.out
#SBATCH -e ISE_train_precomp.err
#SBATCH --time 60:00:00


python3 train.py \
--data_name coco_butd --cnn_type resnet152 --wemb_type glove \
--margin 0.1 --max_violation --img_num_embeds 2 --txt_num_embeds 2 \
--img_attention --txt_attention --img_finetune --txt_finetune \
--mmd_weight 0.01 --div_weight 0.0 --unif_weight 0.1 \
--batch_size 200 --warm_epoch 1 --num_epochs 100 \
--optimizer adamw --lr_scheduler step --lr_step_size 20 --lr_step_gamma 0.1 \
--warm_img --finetune_lr_lower 1 --log_step 500 --lr 1e-3 --txt_lr_scale 1 --img_pie_lr_scale 1 \
--txt_pie_lr_scale 1 --eval_on_gpu --sync_bn --amp --fast_batch \
--loss smooth_chamfer --eval_distance smooth_chamfer --temperature 16 --alpha 0.5 --txt_pooling rnn \
--arch perceiver --txt_attention_head transformer --txt_attention_input wemb \
--perceiver_img_pos_enc_type none --perceiver_txt_pos_enc_type sine \
--perceiver_1x1 --perceiver_residual --perceiver_residual_norm --perceiver_residual_activation sigmoid \
--perceiver_activation gelu \
--perceiver_pre_self_attns 1 --perceiver_query_self_attns 0 \
--perceiver_img_latent_head 8 --perceiver_img_latent_dim 32 \
--perceiver_txt_latent_head 8 --perceiver_txt_latent_dim 32 \
--perceiver_ff_mult 2 \
--perceiver_last_ln --perceiver_last_fc \
--perceiver_cross_head 4 --perceiver_cross_dim 64 \
--img_res_pool max --img_res_first_fc \
--perceiver_input_dim 1024 --perceiver_query_dim 1024 \
--remark coco_butd_alpha0.5_eval_sc --img_1x1_dropout 0.1 --perceiver_pre_norm \
--tri_mean_to_max --gpo_1x1 --gpo_rnn \
--weight_decay 1e-2 --grad_clip 0.1 --lr_warmup 1 --unif_residual \
--workers 4 --dropout 0.1 --caption_drop_prob 0.2 --butd_drop_prob 0.2


