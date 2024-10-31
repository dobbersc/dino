#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/squashfs_test-%j.out


# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/imagenet_2012_train_set_small.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer exec -B /tmp/imagenet_2012_train_set_small.sqfs:/input-data:image-src=/ /home/pml20/dino/containers/finetuning.sif python test.py
