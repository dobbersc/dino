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

# Teach Momentum 0.9
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.9 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.9 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.9.out 2> logs/ms2/teacher-momentum-0.9.err < /dev/null &

# Teach Momentum 0.95
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.95 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.95 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.95.out 2> logs/ms2/teacher-momentum-0.95.err < /dev/null &

# Teach Momentum 0.999
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.999 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.999 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.999.out 2> logs/ms2/teacher-momentum-0.999.err < /dev/null &
