# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.0
# ---

# %% [markdown]
# # Finetuning
#
# - **Backbone Selection**
#   - Options: Vision Transformer (ViT) or ResNet
#   - Load pretrained DINO weights for selected backbone
#
# - **Finetuning Modes**
#   - **Linear Probing**: Freeze backbone layers, train only the classifier head
#   - **Full Fine-tuning**: Update all layers with layer-wise learning rates
#
# - **Dataset Specification**
#   - Supported datasets: CIFAR-10, CIFAR-100, ImageNet
#   - Apply dataset-specific data augmentation and normalization
#
# - **Optimizer Setup**
#   - **Linear Probing**: Higher learning rate on classifier head only
#   - **Full Fine-tuning**: Different learning rates for backbone and classifier
#
# - **Training Loop**
#   - Run training and evaluation over specified epochs
#   - Monitor loss and accuracy for performance assessment
#

# %%
import torch

from dino.datasets import ImageNetDirectoryDataset, regular_transform
from dino.evaluators import LinearEvaluator
from dino.models import HeadType, ModelType, load_model_with_head
from dino.visualize import display_data

# %%
ds_train = ImageNetDirectoryDataset(
    data_dir = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/train",
    transform = regular_transform,
    path_wnids = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/words.txt",
    sample_classes_indices = (28, 163),
    train = True,
)
len(ds_train)

# %%
display_data(ds_train)

# %%
ds_val = ImageNetDirectoryDataset(
    data_dir = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/train",
    transform = regular_transform,
    path_wnids = "/Users/pvmeng/Documents/dino/data/tiny-imagenet-200/words.txt",
    sample_classes_indices = (28, 163),
    train = False,
)
print(len(ds_val))

# %%
display_data(ds_val)

# %%


model = load_model_with_head(
    model_type = ModelType.VIT_DINO_S,
    head_type = HeadType.LINEAR,
    output_dim = 2,
    head_weights = "/Users/pvmeng/Documents/dino/models/head_head.pt",
)

# %%
model.eval()
img, class_idx = ds_val[0]
img_2, class_idx_2 = ds_val[1]
img_3, class_idx_3 = ds_val[2]



predict = lambda imgs: model(torch.stack(imgs)).argmax(dim=1).tolist()
# predict = lambda imgs: [0] * len(imgs)

display_data(ds_val, predict=predict)



# %%
print([class_idx, class_idx_2, class_idx_3])
print(predict([img, img_2, img_3]))

# %%
val_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)

evaluator = LinearEvaluator(val_loader, model)
accuracies = evaluator.evaluate(topk=(1, ))
print(accuracies)
