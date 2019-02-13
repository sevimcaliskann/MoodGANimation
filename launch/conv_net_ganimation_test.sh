#!/usr/bin/env bash


python -m latent_training.conv_net_train \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output_second_try \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_train.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_test.csv \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--batch_size 1 \
--nepochs 1000 \
--is_midfeatures_used 1 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/checkpoints \
--name batch_size_increased \
--image_size 32 \
--load_epoch -1 \
--layers 'ResidualBlock:3' \
--layers 'ResidualBlock:2' \
--is_train 0
