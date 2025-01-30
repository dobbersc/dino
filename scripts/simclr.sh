#!/bin/bash
#SBATCH --job-name=simclr
#SBATCH --partition=gpu-teaching-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/simclr%j.out

SQFS_FILENAME=imagenette2-160.sqfs

# 1. Copy the squashed dataset to the node's /tmp directory
cp /home/space/datasets-sqfs/$SQFS_FILENAME /tmp/

# 2. Define bind target, dataset mount path, and Apptainer image path
DATA_TARGET_DIR=/input-data
DATA_MOUNT="/tmp/$SQFS_FILENAME:$DATA_TARGET_DIR:image-src=/"
APPTAINER_SIF=/home/pml20/dino/containers/dino-dev.sif


COMMAND=(
    "simclr" 
    "data_dir=$DATA_TARGET_DIR/train"
    "evaluator.data_dir=$DATA_TARGET_DIR"
    "evaluator.num_workers=4"
    "batch_size=64" # as high as possible
    "image_size=160" # 224
)

# Combine all elements into a single string
COMMAND_STRING=$(IFS=" "; echo "${COMMAND[*]}")

# 3. Bind the squashed dataset to the Apptainer environment and run the command
export APPTAINER_BINDPATH="${PWD}:/dino"
apptainer run --nv -B "$DATA_MOUNT" "$APPTAINER_SIF" $COMMAND_STRING
