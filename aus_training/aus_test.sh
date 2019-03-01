#!/usr/bin/env bash

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"


CUDA_VISIBLE_DEVICES=$SGE_GPU python -m aus_training.aus_test \
--model_name aus_trainer \
--data_dir /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws \
--train_images_folder cropped_images \
--test_images_folder images \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/training.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/test_ids_small.csv \
--training_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_training.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/aus_test.pkl \
--batch_size 1 \
--nepochs 1000 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name aus_train \
--image_size 128 \
--log_dir /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training \
--load_epoch -1 \
--layers dumb \
--is_train 0
