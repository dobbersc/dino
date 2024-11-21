#!/bin/bash
#SBATCH --job-name=dino_dataset_analysis
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_%A_%a.log         # Standard output and error log
#SBATCH --time=01:00:00
#SBATCH --array=0-2  # Array range (one index per task)

CONTAINER=/home/pml00/dino/containers/dino-dev.sif
SCRIPT="${SLURM_SUBMIT_DIR}/src/dino/dataset_analysis.py"
OUTPUT_DIR="job_results_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_PATH="/tmp/${OUTPUT_DIR}"
mkdir -p "$OUTPUT_PATH"

DATA_TARGET_DIR=/input-data

DATASET_PATHS=(
    "tiny-imagenet-200_train.sqfs"
    "imagenet100_train.sqfs"
    "imagenette2_160_train.sqfs"
)

ARGS_LIST=(
    "--name tiny_imagenet --output-dir ${OUTPUT_PATH} --dataset-path ${DATA_TARGET_DIR} --rows 4 --cols 4"
    "--name imagenet_100  --output-dir ${OUTPUT_PATH} --dataset-path ${DATA_TARGET_DIR} --rows 4 --cols 4"
    "--name imagenette --output-dir ${OUTPUT_PATH} --dataset-path ${DATA_TARGET_DIR} --rows 4 --cols 4"
)

SCRIPT_ARGS="${ARGS_LIST[$SLURM_ARRAY_TASK_ID]}"
DATASET_PATH="${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}"

cp /home/space/datasets-sqfs/$DATASET_PATH /tmp/

DATA_MOUNT="/tmp/$DATASET_PATH:$DATA_TARGET_DIR:image-src=/"

echo "Running task $SLURM_ARRAY_TASK_ID"

apptainer exec --bind "$DATA_MOUNT" "$CONTAINER" python "$SCRIPT" $SCRIPT_ARGS

cd /tmp/
tar -cf zz_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar "$OUTPUT_DIR"
cp zz_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar $SLURM_SUBMIT_DIR
rm -rf "OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo "Execution successful for dataset: $DATASET_PATH."
else
    echo "Execution failed for dataset: $DATASET_PATH."
fi