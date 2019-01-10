#!/usr/bin/env bash

python /scratch_net/zinc/csevim/apps/repos/GANimation/train.py \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--images_folder images \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/train_ids_whole.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/test_ids_whole.csv \
--aus_file - \
--aus_folder /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/dataset \
--name experiment_2 \
--batch_size 16 \
