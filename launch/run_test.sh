#!/usr/bin/env bash

python /scratch_net/zinc/csevim/apps/repos/GANimation/test.py \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--images_folder images \
--input_path /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/images/N_0000001149_00039.jpg \
--output_dir /scratch_net/zinc/csevim/apps/repos/GANimation/test_outputs \
--aus_file - \
--aus_folder /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/dataset \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name experiment_2_small \
--load_epoch -1
