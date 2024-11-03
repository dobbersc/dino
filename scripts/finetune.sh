#!/bin/bash
#SBATCH --job-name=dino-finetuning
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/finetuning-%j.out

SQFS_FILENAME=imagenet_2012_train_set_small.sqfs

# 1. Copy the squashed dataset to the node's /tmp directory
cp /home/space/datasets-sqfs/$SQFS_FILENAME /tmp/

# 2. Define bind target, dataset mount path, and Apptainer image path
DATA_TARGET_DIR=/input-data
DATA_MOUNT="/tmp/$SQFS_FILENAME:$DATA_TARGET_DIR:image-src=/"
APPTAINER_SIF=/home/pml20/dino/containers/finetuning.sif
COMMAND="dino finetune --dataset imagenet"
# COMMAND="python scripts/test.py"

# 3. Bind the squashed dataset to the Apptainer environment and run the command
export APPTAINER_BINDPATH="${PWD}:/dino"
apptainer run --nv -B "$DATA_MOUNT" "$APPTAINER_SIF" $COMMAND
