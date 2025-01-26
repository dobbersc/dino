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

# Number Local Views (1)
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-1 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_num_views=1 \
  > logs/ms3/number-local-views-1.out 2> logs/ms3/number-local-views-1.err < /dev/null &

# Number Local Views (2)
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-2 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_num_views=2 \
  > logs/ms3/number-local-views-2.out 2> logs/ms3/number-local-views-2.err < /dev/null &

# Number Local Views (4)
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-4 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_num_views=4 \
  > logs/ms3/number-local-views-4.out 2> logs/ms3/number-local-views-4.err < /dev/null &

# Number Local Views (8)
# Covered by default parameter run
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-8 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_num_views=8 \
  > logs/ms3/number-local-views-8.out 2> logs/ms3/number-local-views-8.err < /dev/null &

# Number Local Views (16)
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-16 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  local_augmenter_num_views=16 \
  > logs/ms3/number-local-views-16.out 2> logs/ms3/number-local-views-16.err < /dev/null &
