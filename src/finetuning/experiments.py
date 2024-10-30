# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
from torchvision import models
from transformers import ViTModel
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def load_backbone(model_type='vit', pretrained_dino_weights=None):
    if model_type == 'vit':
        model = ViTModel.from_pretrained(pretrained_dino_weights)  # Load pretrained DINO weights for ViT
    elif model_type == 'resnet':
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(pretrained_dino_weights))
    return model


num_classes = 10

def setup_finetuning_mode(model, finetune_mode='linear_probing'):
    if finetune_mode == 'linear_probing':
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)  # Replace head
    elif finetune_mode == 'full_finetuning':
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)  # Replace head
        # Optionally set up a per-layer learning rate scheduler in the optimizer
    return model


def load_dataset(dataset_name='cifar10'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    elif dataset_name == 'imagenet':
        dataset = datasets.ImageNet(root='./data', split='train', transform=transform)
    return dataset




def setup_optimizer(model, finetune_mode='linear_probing', base_lr=1e-3):
    if finetune_mode == 'linear_probing':
        optimizer = optim.Adam(model.classifier.parameters(), lr=base_lr)
    elif finetune_mode == 'full_finetuning':
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': model.classifier.parameters(), 'lr': base_lr}
        ])
    return optimizer


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


