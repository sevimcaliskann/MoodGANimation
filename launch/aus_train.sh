#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=$SGE_GPU python -m aus_training.aus_training \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--train_images_folder cropped_images \
--test_images_folder images \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/training.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/test_ids_small.csv \
--training_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_training.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_test.pkl \
--batch_size 16 \
--nepochs 1000 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name aus_train \
--load_epoch -1
