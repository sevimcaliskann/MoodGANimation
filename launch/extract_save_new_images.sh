#!/usr/bin/env bash

python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/from_network_with_labels \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name affectnet_mood_default10 \
--cond_nc 3 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_latent_inception.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl


python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/from_network_4d \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name affectnet_mood_default12 \
--cond_nc 4 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/generative_sampling.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_4d_latent_inception.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl
