#!/usr/bin/env bash

python /scratch_net/zinc/csevim/apps/repos/GANimation/get_middle_layer_features.py \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat \
--images_folder imgs \
--input_file emotion_cat_all.csv \
--output_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output_all_zero \
--output_file features.pkl \
--aus_file aus_emotion_cat.pkl \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name amplified_cycle_loss_04_02 \
--cond_nc 17 \
--layers 'ResidualBlock:3' \
--layers 'ResidualBlock:2' \
--layers 'attention' \
--load_epoch -1 \
--gpu_ids $SGE_GPU
