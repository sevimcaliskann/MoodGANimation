#!/usr/bin/env bash

python /scratch_net/zinc/csevim/apps/repos/GANimation/test.py \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--training_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_training.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_test.pkl \
--aus_csv_folder /scratch_net/zinc/csevim/apps/repos/GANimation/faces/openface_out \
--input_path /scratch_net/zinc/csevim/apps/repos/GANimation/faces/imgs/face.png \
--output_dir /scratch_net/zinc/csevim/apps/repos/GANimation/test_outputs \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name amplified_cycle_loss_04_02 \
--cond_nc 17 \
--load_epoch -1 \
--au_index 0
#--gpu_ids 2

#--name middle_mask_loss_01_02 \
