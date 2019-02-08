#!/usr/bin/env bash


python latent_training/svm_training.py \
--data_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/output \
--imgs_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/cropped_imgs \
--ids_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/emotion_cat_all.csv \
--labels_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/emotion_cat_aws.xlsx \
--save_folder /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/latent_training \
--batch_size 16 \
--nepochs 1000 \
--is_midfeatures_used -1
