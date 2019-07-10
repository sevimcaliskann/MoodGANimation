#!/usr/bin/env bash

python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/affectnet_au \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name ganimation_baseline_on_affectnet \
--cond_nc 17 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/aus_affectnet.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl


'''python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/maximize_mask_no_cycle \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name maximize_mask_no_cycle \
--cond_nc 4 \
--load_epoch 50 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train2_norm_1000_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl


python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/maximize_mask \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name maximize_mask \
--cond_nc 4 \
--load_epoch 33 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_norm_1000_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl


python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/ganimation_adv_100 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name ganimation_adv_100 \
--cond_nc 4 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_norm_1000_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl'''


'''python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/no_cycle_high_adv_high_cond \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name no_cycle_high_adv_high_cond \
--cond_nc 4 \
--load_epoch 27 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train2_norm_1000_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl

python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/no_cycle_5d \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name no_cycle_5d \
--cond_nc 5 \
--load_epoch 27 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_5d_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl


python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/no_cycle_2d_resnet50 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name no_cycle_2d_resnet50 \
--cond_nc 2 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_2d_3fully_resnet50.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl'''
