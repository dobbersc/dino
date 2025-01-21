# Augmentations: Crop Only
nohup train \
  experiment_tag=ms3 \
  model_tag=augmentations-crop-only \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  augmentation_pipeline=CROP_ONLY \
  > logs/ms3/augmentations-crop-only.out 2> logs/ms3/augmentations-crop-only.err < /dev/null &

# Local View Crop Scale Interval (0.05, 0.10)
nohup train \
  experiment_tag=ms3 \
  model_tag=local-view-crop-scale-0.05-0.10 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_crop_scale="[0.05, 0.10]" \
  > logs/ms3/local-view-crop-scale-0.05-0.10.out 2> logs/ms3/local-view-crop-scale-0.05-0.10.err < /dev/null &

# Local View Crop Scale Interval (0.20, 0.25)
nohup train \
  experiment_tag=ms3 \
  model_tag=local-view-crop-scale-0.20-0.25 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_crop_scale="[0.20, 0.25]" \
  > logs/ms3/local-view-crop-scale-0.20-0.25.out 2> logs/ms3/local-view-crop-scale-0.20-0.25.err < /dev/null &

# Local View Crop Scale Interval (0.35, 0.40)
nohup train \
  experiment_tag=ms3 \
  model_tag=local-view-crop-scale-0.35-0.40 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_crop_scale="[0.35, 0.40]" \
  > logs/ms3/local-view-crop-scale-0.35-0.40.out 2> logs/ms3/local-view-crop-scale-0.35-0.40.err < /dev/null &
