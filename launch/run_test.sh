#!/usr/bin/env bash

python test.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--training_aus_file /srv/glusterfs/csevim/datasets/affectnet/aus_affectnet.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/affectnet/aus_affectnet.pkl \
--input_path /home/sevim/Downloads/faces/imgs/face5.jpg \
--output_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/test_outputs \
--checkpoints_dir /home/sevim/Downloads/master_thesis_study_documents/code-examples/ganimation_checkpoints \
--name affwild_4d \
--comparison_model_name affwild_4d_nonrecurrent \
--cond_nc 4 \
--load_epoch -1 \
--comparison_load_epoch -1 \
--frames_cnt 17 \
--moods_pickle_file /home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/annotations/255_4d.pkl \
--groundtruth_video /home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/videos/255.mp4 \
--recurrent True
#--gpu_ids 2

#--name middle_mask_loss_01_02 \
