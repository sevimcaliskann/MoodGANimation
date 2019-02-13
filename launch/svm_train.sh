#!/usr/bin/env bash


python -m latent_training.svm_training \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_all.csv \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--save_folder /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models \
--batch_size 16 \
--nepochs 1000 \
--is_midfeatures_used -1 \
--is_train 0 
