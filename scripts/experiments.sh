# Architecture DeiT-Small
nohup train \
  experiment_tag=ms2 \
  model_tag=architecture-deit-small \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=DEIT_S \
  > logs/ms2/architecture-deit-small.out 2> logs/ms2/architecture-deit-small.err < /dev/null &

# Architecture ResNet50
nohup train \
  experiment_tag=ms2 \
  model_tag=architecture-resnet50 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=RESNET_50 \
  > logs/ms2/architecture-resnet50.out 2> logs/ms2/architecture-resnet50.err < /dev/null &

# Model Collapse No Sharpening
nohup train \
  experiment_tag=ms2 \
  model_tag=model-collapse-no-sharpening \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_temperature_initial=1 \
  teacher_temperature_final=null \
  > logs/ms2/model-collapse-no-sharpening.out 2> logs/ms2/model-collapse-no-sharpening.err < /dev/null &

# Model Collapse No Centering
nohup train \
  experiment_tag=ms2 \
  model_tag=model-collapse-no-centering \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  center_momentum=null \
  > logs/ms2/model-collapse-no-centering.out 2> logs/ms2/model-collapse-no-centering.err < /dev/null &
