#!/usr/bin/env bash

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"


CUDA_VISIBLE_DEVICES=$SGE_GPU python -m aus_training.aus_with_rule_learning \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_train.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_test.csv \
--aus_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/aus_emotion_cat.pkl \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--batch_size 1 \
--nepochs 1000 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name fully_connect_weightless_cross_rule_learning \
--rule_learning 1 \
--load_epoch -1 \
--is_train 0 \
--imgs_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/cropped_imgs \
--image_size 128 \
--log_dir /scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training \
--layers dumb
