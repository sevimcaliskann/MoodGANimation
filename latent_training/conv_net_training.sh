#!/usr/bin/env bash


python -m latent_training.conv_net_train \
--data_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/output \
--imgs_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/cropped_imgs \
--train_ids_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/emotion_cat_train.csv \
--test_ids_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/emotion_cat_test.csv \
--labels_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/emotion_cat_aws.xlsx \
--batch_size 16 \
--nepochs 100 \
--is_midfeatures_used 1 \
--checkpoints_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/checkpoints \
--name batch_size_increased \
--image_size 32 \
--load_epoch -1
