#!/usr/bin/env bash


python -m latent_training.fully_connected_train \
--train_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_train.csv \
--test_ids_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_test.csv \
--aus_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/aus_emotion_cat.pkl \
--labels_file /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/emotion_cat_aws.xlsx \
--batch_size 1 \
--nepochs 1000 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/emotion_cat/checkpoints \
--name baseline \
--rule_learning 0 \
--load_epoch -1 \
--is_train 0
