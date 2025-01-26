import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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
    LOG_STEPS = 100

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

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        logger.info("Start SimCLR training for %d epochs.", self.epochs)

        for epoch_counter in range(self.epochs):
            for images in tqdm(train_loader):
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

                if n_iter % self.LOG_STEPS == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    logger.info("Step %d: Loss = %.10f", n_iter, loss)
                    logger.info("Step %d: Top-1 Accuracy = %.2f%%", n_iter, top1[0])
                    logger.info("Step %d: Top-5 Accuracy = %.2f%%", n_iter, top5[0])
                    learning_rate = self.scheduler.get_lr()[0]
                    logger.info("Step %d: Learning Rate = %.6f", n_iter, learning_rate)

                n_iter += 1

            metrics = {
                "loss": loss.item(),
                "top1-accuracy": top1[0].item(),
                "top5-accuracy": top5[0].item(),
                "learning-rate": learning_rate,
            }

            if self.evaluator is not None:
                eval_results = self.evaluator(self.model)
                metrics.update(eval_results)
                logger.info("Step %d: Evaluation results: %s", n_iter, eval_results)

            mlflow_log_metrics(
                "train_batch",
                metrics=metrics,
                step=epoch_counter,
            )
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logger.debug(
                "Epoch: %d\tLoss: %.4f\tTop1 accuracy: %.2f%%",
                epoch_counter,
                loss,
                top1[0],
            )
