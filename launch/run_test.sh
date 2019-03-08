#!/usr/bin/env bash


export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"

python /scratch_net/zinc/csevim/apps/repos/GANimation/test.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--training_aus_file /srv/glusterfs/csevim/datasets/affectnet/aus_affectnet.pkl \
--test_aus_file /srv/glusterfs/csevim/datasets/affectnet/aus_affectnet.pkl \
--input_path /scratch_net/zinc/csevim/apps/repos/GANimation/faces/imgs/face.png \
--output_dir /scratch_net/zinc/csevim/apps/repos/GANimation/test_outputs \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotionet/checkpoints \
--name affectnet_with_moods \
--cond_nc 13 \
--load_epoch -1 \
--gpu_ids $SGE_GPU
#--gpu_ids 2

#--name middle_mask_loss_01_02 \
