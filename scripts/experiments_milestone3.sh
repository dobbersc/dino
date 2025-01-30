# SimCLR DeiT-Small
nohup simclr \
  experiment_tag=ms3-simclr \
  model_tag=simclr-deit-small \
  data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  num_workers=8 \
  evaluator.data_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  evaluator.num_workers=8 \
  batch_size=256 \
  epochs=200 \
  > logs/ms3/simclr-deit-small.out 2> logs/ms3/simclr-deit-small.err < /dev/null &

# Architecture DeiT-Small Evaluation
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=DEIT_S \
  backbone.weights=models/simclr-deit-small.pt \
  k=5 \
  finetuning_mode=LINEAR_PROBE \
  num_epochs=25 \
  batch_size=256 \
  base_lr=0.001 \
  > logs/ms3/simclr-deit-small-evaluation.out 2> logs/ms3/simclr-deit-small-evaluation.err < /dev/null &

# SimCLR ResNet50
nohup simclr \
  experiment_tag=ms3-simclr \
  model_tag=simclr-resnet50 \
  data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  num_workers=8 \
  evaluator.data_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  evaluator.num_workers=8 \
  batch_size=256 \
  epochs=100 \
  backbone.model_type=RESNET_50 \
  > logs/ms3/simclr-resnet50.out 2> logs/ms3/simclr-resnet50.err < /dev/null &

# Architecture DeiT-Small Evaluation
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=RESNET_50 \
  backbone.weights=models/simclr-resnet50.pt \
  k=5 \
  finetuning_mode=LINEAR_PROBE \
  num_epochs=25 \
  batch_size=256 \
  base_lr=0.001 \
  > logs/ms3/simclr-resnet50-evaluation.out 2> logs/ms3/simclr-resnet50-evaluation.err < /dev/null &

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

# Number Local Views (16) Less Epochs
nohup train \
  experiment_tag=ms3 \
  model_tag=number-local-views-16-e50 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  max_epochs=50 \
  local_augmenter_num_views=16 \
  > logs/ms3/number-local-views-16-e50.out 2> logs/ms3/number-local-views-16-e50.err < /dev/null &
