#!/usr/bin/env bash

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"

CUDA_VISIBLE_DEVICES=$SGE_GPU python extract_images.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--test_images_folder cropped2 \
--output_dir /srv/glusterfs/csevim/datasets/fid_test/from_network_with_labels \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--name affectnet_mood_default10 \
--cond_nc 3 \
--load_epoch 30 \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/test_mood.csv \
--moods_pickle_file /srv/glusterfs/csevim/datasets/affectnet/train_latent_inception.pkl \
--emo_test_file /srv/glusterfs/csevim/datasets/affectnet/affectnet_emos.pkl
