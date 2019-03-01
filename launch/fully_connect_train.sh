#!/usr/bin/env bash

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"

CUDA_VISIBLE_DEVICES=$SGE_GPU python -m latent_training.fully_connected_train \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_train.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_test.csv \
--aus_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/aus_emotion_cat.pkl \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--batch_size 16 \
--nepochs 1000 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/checkpoints \
--name fully_connect_weightless_cross_rule_learning_val_train_acc \
--rule_learning 1 \
--load_epoch -1 \
--is_train 1
