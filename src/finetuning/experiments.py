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
from finetuning.datasets import ImageNetDirectoryDataset
import os
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import islice
import random

# %%
# cwd

imagenet_path = Path(os.getcwd()).parent.parent / "data" / "tiny-imagenet-200" / "train"
wnid_path = Path(os.getcwd()).parent.parent / "data" / "tiny-imagenet-200" / "words.txt"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224 for ImageNet models
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet normalization
    ]
)
dataset = ImageNetDirectoryDataset(
    imagenet_path, transform=transform, path_wnids=wnid_path, num_sample_classes=5
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


NUM_CLASSES = len(dataset.class_idx_to_wnid)
NUM_BATCHES = len(dataloader)
NUM_SAMPLES = len(dataset)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Number of samples: {NUM_SAMPLES}")
print(f"Number of batches: {NUM_BATCHES}")


# %%
def unnorm(img):
    # Unnormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std[:, None, None] + mean[:, None, None]  # Unnormalize
    return img.clamp(0, 1)


def display_data(dataset, class_idx=None, predict=None):
    NUM_IMAGES = 12
    # Assuming `dataset.get_image_by_class` is a generator
    max_length = 10
    if class_idx is not None:
        raw_images = list(islice(dataset.get_image_by_class(class_idx), NUM_IMAGES))
        class_indices = [class_idx] * NUM_IMAGES
        titles = None
    else:
        # pick random images
        img_indices = random.sample(range(len(dataset)), NUM_IMAGES)
        raw_images, class_indices = zip(*[dataset[i] for i in img_indices])
        titles = [f"g={dataset.get_class_name(g)[:max_length]}" for g in class_indices]

    if predict is not None:
        # Assuming `predict` is a function that takes in a list of images and returns a list of predictions
        predictions = predict(raw_images)
        titles = [
            f"y={dataset.get_class_name(y)[:max_length]}\ng={dataset.get_class_name(g)[:max_length]}"
            for y, g in zip(predictions, class_indices)
        ]

    images = [unnorm(img) for img in raw_images]

    # display 4x4
    fig, ax = plt.subplots(2, 6, figsize=(12, 4))
    for i, (img, ax) in enumerate(zip(images, ax.flatten())):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])

    if titles is None:
        title = dataset.get_class_name(class_idx)
        fig.suptitle(title)
    plt.show()


# %%
def predict_random(images):
    # Randomly predict the class of each image
    return [random.randint(0, NUM_CLASSES - 1) for _ in images]

display_data(dataset, predict=predict_random)


# %%
def load_backbone(model_type="vit", pretrained_dino_weights=None):
    if model_type == "vit":
        if pretrained_dino_weights is None:
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        else:
            raise NotImplementedError(
                "Loading pretrained weights for ViT models is not yet supported"
            )

    elif model_type == "resnet":
        if pretrained_dino_weights is None:
            model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        else:
            raise NotImplementedError(
                "Loading pretrained weights for ViT models is not yet supported"
            )

    return model


# %%
class VisionTransformerWithLinearHead(nn.Module):
    def __init__(self, model, embed_dim, num_classes):
        super().__init__()
        self.model = model
        self.head = nn.Linear(embed_dim, num_classes)

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self):
        return self.model.embed_dim

    @property
    def num_classes(self):
        return self.head.out_features

    def backbone_paramters(self):
        return self.model.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, x):
        x = self.model(x)
        return self.head(x)


def setup_finetuning_mode(model, embed_dim, num_classes, freeze_backbone=True):
    headed_model = VisionTransformerWithLinearHead(model, embed_dim, num_classes)
    if freeze_backbone:
        headed_model.freeze_backbone()
    return headed_model


# %%
model = load_backbone()

model = setup_finetuning_mode(model, embed_dim=model.embed_dim, num_classes=NUM_CLASSES)
_ = model.eval()


# %%
batch_1 = dataset[0][0].unsqueeze(0)

res = model(batch_1)

# get the class index with the highest score
class_idx = torch.argmax(res).item()
class_idx

# %%
base_lr = 1e-3

optimizer = optim.Adam(model.head_parameters(), lr=base_lr)

optimizer = optim.AdamW(
    [
        {"params": model.backbone_paramters(), "lr": base_lr * 0.1},
        {"params": model.head_parameters(), "lr": base_lr},
    ]
)


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        count = 0
        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")



# %%
train(model, dataloader, nn.CrossEntropyLoss(), optimizer, num_epochs=1)

# %%
# save the model
torch.save(model.state_dict(), "model.pth")



# %%
# load the model
model = load_backbone()
model = setup_finetuning_mode(model, embed_dim=model.embed_dim, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("model.pth"))

# %%
_ = model.eval()

model_predict = lambda lst: model(torch.stack(lst)).argmax(dim=1).tolist()

display_data(dataset, predict=model_predict)
