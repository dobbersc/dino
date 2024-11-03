from enum import Enum

from torch import optim
from tqdm import tqdm


class FinetuningMode(Enum):
    LINEAR_PROBE = "linear_probe"
    FULL_FINETUNE = "full_finetune"


def get_optimizer(model, base_lr=1e-3, mode=FinetuningMode.LINEAR_PROBE):
    if mode == FinetuningMode.LINEAR_PROBE:
        optimizer = optim.Adam(model.head_parameters(), lr=base_lr)
    elif mode == FinetuningMode.FULL_FINETUNE:
        optimizer = optim.AdamW(
            [
                {"params": model.backbone_paramters(), "lr": base_lr * 0.1},
                {"params": model.head_parameters(), "lr": base_lr},
            ]
        )
    return optimizer


def train(model, dataloader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


def fine_tune(
    model, dataloader, criterion, base_lr=1e-3, mode=FinetuningMode.LINEAR_PROBE, num_epochs=10, device="cpu"
):
    if mode == FinetuningMode.LINEAR_PROBE:
        model.freeze_backbone()
    optimizer = get_optimizer(model, base_lr, mode)
    train(model, dataloader, criterion, optimizer, num_epochs, device=device)
