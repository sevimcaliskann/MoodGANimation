#!/usr/bin/env bash


'''python -m latent_training.svm_training \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_all.csv \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--save_folder /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models/au_from_emotionet \
--kfold 10 \
--batch_size 16 \
--nepochs 1000 \
--input_mode au \
--is_train 1'''


'''python -m latent_training.svm_training \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_all.csv \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--save_folder /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models/k_fold_test \
--kfold 10 \
--batch_size 16 \
--nepochs 1000 \
--input_mode res \
--is_train 1'''


python -m latent_training.svm_training \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/output_all_zero \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_all.csv \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--save_folder /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models/epoch_100_res_with_zero_aus \
--kfold 1 \
--batch_size 16 \
--nepochs 100 \
--input_mode res \
--randomize_aus 1 \
--is_train 1
