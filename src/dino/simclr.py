import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dino import LOG_SEPARATOR
from dino.utils.logging import mlflow_log_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SimCLR:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        n_views,
        batch_size,
        temperature,
        epochs,
        fp16_precision=False,
        evaluator: Callable[[torch.nn.Module], dict[str, float]] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.n_views = n_views
        self.batch_size = batch_size
        self.temperature = temperature
        self.epochs = epochs
        self.fp16_precision = fp16_precision
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.evaluator = evaluator

    def info_nce_loss(self, features):
        labels = torch.cat(
            [torch.arange(self.batch_size) for i in range(self.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.n_views * self.batch_size,
            self.n_views * self.batch_size,
        )
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.fp16_precision)

        logger.info(LOG_SEPARATOR)
        logger.info("Start SimCLR training for %d epochs.", self.epochs)
        logger.info(LOG_SEPARATOR)
        logger.info(self.model)

        n_iter = 1  # Total step counter
        for epoch_counter in tqdm(range(1, self.epochs + 1)):
            logger.info(LOG_SEPARATOR)
            logger.info("EPOCH %d; STEP %d", epoch_counter, n_iter)

            self.model.train()  # Set model into training mode at the beginning of each epoch

            epoch_loss: float = 0.0
            epoch_top1_accuracy: float = 0.0
            epoch_top5_accuracy: float = 0.0
            batch_index: int = 0

            for images in tqdm(train_loader, unit="batch", leave=False):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast(enabled=self.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # Log step results
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                learning_rate = self.optimizer.param_groups[0]["lr"]
                mlflow_log_metrics(
                    "train_batch",
                    metrics={
                        "epoch": epoch_counter,
                        "loss": loss,
                        "learning_rate": learning_rate,
                        "top1_train_accuracy": top1.item() / 100,
                        "top5_train_accuracy": top5.item() / 100,
                    },
                    step=n_iter,
                )

                epoch_loss += loss.item()
                epoch_top1_accuracy += top1.item()
                epoch_top5_accuracy += top5.item()

                if batch_index % max(1, len(train_loader) // 10) == 0:
                    msg: str = (
                        f"batch {batch_index + 1}/{len(train_loader)}"
                        f" - step {n_iter}"
                        f" - loss {loss:.8}"
                        f" - top1 train accuracy {top1.item():.2f}%"
                        f" - top5 train accuracy {top5.item():.5f}%"
                        f" - lr {learning_rate:.8f}"
                    )
                    logger.info(msg)

                n_iter += 1
                batch_index += 1

            logger.info("EPOCH %d DONE (step %d)", epoch_counter, n_iter)

            # Average running epoch metrics
            epoch_loss /= len(train_loader)
            epoch_top1_accuracy /= len(train_loader)
            epoch_top5_accuracy /= len(train_loader)

            metrics = {
                "loss": epoch_loss,
                "top1_train_accuracy": epoch_top1_accuracy / 100,
                "top5_train_accuracy": epoch_top5_accuracy / 100,
            }

            # Evaluate model at the end of the epoch
            if self.evaluator is not None:
                logger.info(LOG_SEPARATOR)
                logger.info("Evaluation Phase")
                eval_results = self.evaluator(self.model)
            else:
                eval_results = {}

            logger.info(LOG_SEPARATOR)
            logger.info("Loss: %.8f", epoch_loss)
            logger.info("Top-1 Train Accuracy: %.2f%%", epoch_top1_accuracy)
            logger.info("Top-5 Train Accuracy: %.2f%%", epoch_top5_accuracy)
            if eval_results:
                logger.info("Evaluation Results: %r", eval_results)

            # Log epoch metrics
            mlflow_log_metrics("train_epoch", metrics=metrics | eval_results, step=epoch_counter)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
