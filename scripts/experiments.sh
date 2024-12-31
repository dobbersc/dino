# Architecture DeiT-Small
nohup train \
  experiment_tag=ms2 \
  model_tag=architecture-deit-small \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=DEIT_S \
  > logs/ms2/architecture-deit-small.out 2> logs/ms2/architecture-deit-small.err < /dev/null &

# Architecture DeiT-Small Evaluation
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=DEIT_S \
  backbone.weights=models/architecture-deit-small-student.pt \
  k=5 \
  finetuning_mode=LINEAR_PROBE \
  num_epochs=25 \
  batch_size=256 \
  base_lr=0.001 \
  > logs/ms2/architecture-deit-small-student-evaluation.out 2> logs/ms2/architecture-deit-small-student-evaluation.err < /dev/null &

# Architecture ResNet50
nohup train \
  experiment_tag=ms2 \
  model_tag=architecture-resnet50 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=RESNET_50 \
  > logs/ms2/architecture-resnet50.out 2> logs/ms2/architecture-resnet50.err < /dev/null &

# Architecture ResNet50 Evaluation
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=RESNET_50 \
  backbone.weights=models/architecture-resnet50-student.pt \
  k=5 \
  finetuning_mode=LINEAR_PROBE \
  num_epochs=25 \
  batch_size=256 \
  base_lr=0.001 \
  > logs/ms2/architecture-resnet50-student-evaluation.out 2> logs/ms2/architecture-resnet50-student-evaluation.err < /dev/null &

# Architecture DeiT-Small Fully Supervised
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  skip_knn=True \
  finetuning_mode=FULL_FINETUNE \
  backbone.model_type=DEIT_S \
  num_epochs=100 \
  batch_size=128 \
  base_lr=5e-4 \
  backbone_lr=5e-4 \
  model_dir=models \
  model_tag=architecture-deit-small-supervised \
  > logs/ms2/architecture-deit-small-supervised.out 2> logs/ms2/architecture-deit-small-supervised.err < /dev/null &
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=DEIT_S \
  backbone.weights=models/architecture-deit-small-supervised-0.0005-128-backbone.pt \
  k=5 \
  skip_linear=True \
  > logs/ms2/architecture-deit-small-supervised-knn.out 2> logs/ms2/architecture-deit-small-supervised-knn.err < /dev/null &

# Architecture ResNet50 Fully Supervised
# Learning rate uses linear scaling rule: 0.001 * batch_size / 256.
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  skip_knn=True \
  finetuning_mode=FULL_FINETUNE \
  backbone.model_type=RESNET_50 \
  num_epochs=100 \
  batch_size=128 \
  base_lr=5e-4 \
  backbone_lr=5e-4 \
  model_dir=models \
  model_tag=architecture-resnet50-supervised \
  > logs/ms2/architecture-resnet50-supervised.out 2> logs/ms2/architecture-resnet50-supervised.err < /dev/null &
nohup evaluate \
  dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  backbone.model_type=RESNET_50 \
  backbone.weights=models/architecture-resnet50-supervised-0.0005-128-backbone.pt \
  k=5 \
  skip_linear=True \
  > logs/ms2/architecture-resnet50-supervised-knn.out 2> logs/ms2/architecture-resnet50-supervised-knn.err < /dev/null &


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

# Teacher Momentum 0.9
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.9 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.9 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.9.out 2> logs/ms2/teacher-momentum-0.9.err < /dev/null &

# Teacher Momentum 0.95
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.95 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.95 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.95.out 2> logs/ms2/teacher-momentum-0.95.err < /dev/null &

# Teacher Momentum 0.99
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.99 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.99 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.99.out 2> logs/ms2/teacher-momentum-0.99.err < /dev/null &

# Teacher Momentum 0.999
nohup train \
  experiment_tag=ms2 \
  model_tag=teacher-momentum-0.999 \
  dataset.data_dir=/vol/tmp/dobbersc-pub/imagenet100/train \
  evaluation_dataset_dir=/vol/tmp/dobbersc-pub/imagenet100 \
  teacher_momentum_initial=0.999 \
  teacher_momentum_final=null \
  > logs/ms2/teacher-momentum-0.999.out 2> logs/ms2/teacher-momentum-0.999.err < /dev/null &
