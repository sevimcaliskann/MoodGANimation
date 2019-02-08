#!/usr/bin/env bash

python get_middle_layer_features.py \
--data_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat \
--images_folder imgs \
--input_file emotion_cat_all.csv \
--output_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/emotion_cat/output \
--output_file features.pkl \
--aus_file aus_emotion_cat.pkl \
--checkpoints_dir checkpoints \
--name mask_reduced_cropped_30_01 \
--cond_nc 17 \
--load_epoch -1
