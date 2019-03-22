#!/usr/bin/env bash

python test.py \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--training_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_training.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_test.pkl \
--aus_csv_folder /scratch_net/zinc/csevim/apps/repos/GANimation/faces/openface_out \
--input_path /home/sevim/Downloads/faces/imgs/face.png \
--output_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/test_outputs \
--checkpoints_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/ganimation_checkpoints/ \
--name ganimation_baseline_on_affectnet \
--cond_nc 17 \
--load_epoch -1 \
--au_index 8
#--gpu_ids 2

#--name middle_mask_loss_01_02 \
