from typing import TYPE_CHECKING

import timm
from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from dino.datasets import UnlabelledDataset
from dino.models import DINOHead, ModelWithHead
from dino.trainer import DINOTrainer
from dino.utils.random import set_seed
from dino.utils.torch import detect_device

if TYPE_CHECKING:
    from timm.models import VisionTransformer
    from torch import Tensor
    from torch.utils.data import Dataset


def train() -> None:
    set_seed(42)

    head_hidden_dim: int = 2028
    head_output_dim: int = 1024
    # dataset_dir: str = "/vol/tmp/dobbersc-pub/imagenette2/train" # noqa: ERA001
    dataset_dir: str = "/vol/tmp/dobbersc-pub/tiny-imagenet-200/train"
    batch_size: int = 128

    student: VisionTransformer = timm.create_model("deit_small_patch16_224", num_classes=0, dynamic_img_size=True)
    teacher: VisionTransformer = timm.create_model("deit_small_patch16_224", num_classes=0, dynamic_img_size=True)

    student_with_head: ModelWithHead = ModelWithHead(
        model=student,
        head=DINOHead(input_dim=student.num_features, output_dim=head_output_dim, hidden_dim=head_hidden_dim),
    )
    teacher_with_head: ModelWithHead = ModelWithHead(
        model=teacher,
        head=DINOHead(input_dim=teacher.num_features, output_dim=head_output_dim, hidden_dim=head_hidden_dim),
    )

    teacher_with_head.load_state_dict(student_with_head.state_dict())

    dataset: Dataset[Tensor] = UnlabelledDataset(dataset=ImageFolder(dataset_dir, transform=v2.ToImage()))

    trainer: DINOTrainer = DINOTrainer(student=student_with_head, teacher=teacher_with_head, dataset=dataset)
    trainer.train(
        max_epochs=100,
        batch_size=batch_size,
        loss_function_kwargs={"output_size": head_output_dim},
        device=detect_device(),
        optimizer_class=AdamW,
        optimizer_kwargs={
            "lr": 0.0005 * batch_size / 256,
            "weight_decay": 0.04,  # Max: 0.4 TODO: Scheduler this as well
        },
        num_workers=8,
    )
