import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nltk  # type: ignore[import-untyped]
import timm
import torch
from adjustText import adjust_text  # type: ignore[import-untyped]
from nltk.corpus import wordnet  # type: ignore[import-untyped]
from PIL import Image
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from timm.models import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # type: ignore[import-untyped]
from tqdm import tqdm

from dino.augmentation import DefaultGlobalAugmenter, DefaultLocalAugmenter
from dino.utils.torch import detect_device

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

device = detect_device()


def _normalize_img(img: torch.Tensor) -> torch.Tensor:
    img = img.clone()
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    return img.to(torch.uint8)


def plot_original_image(image: torch.Tensor, output_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(_normalize_img(image.permute(1, 2, 0)).numpy())
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(output_dir / "original_image.pdf")


def plot_augmentations(image: torch.Tensor, output_dir: Path) -> None:
    repeats = 8
    global_augmenter = DefaultGlobalAugmenter()
    local_augmenter = DefaultLocalAugmenter(repeats=repeats)
    global_views = global_augmenter(image)
    local_views = local_augmenter(image)
    fig, axs = plt.subplots(nrows=3, ncols=4)
    flattened_axs = axs.reshape(-1)
    for i, gv in enumerate(global_views):
        flattened_axs[i + 1].imshow(_normalize_img(gv.permute(1, 2, 0)).numpy())
        flattened_axs[i + 1].axis("off")
        flattened_axs[i + 1].set_title(f"Global View {i+1}")
    fig.delaxes(flattened_axs[0])
    fig.delaxes(flattened_axs[3])

    for i, lv in enumerate(local_views):
        flattened_axs[i + 4].imshow(_normalize_img(lv.permute(1, 2, 0)).numpy())
        flattened_axs[i + 4].axis("off")
        flattened_axs[i + 4].set_title(f"Local View {i+1}")
    fig.tight_layout()
    plt.savefig(output_dir / "augmentations.pdf")


def plot_attention(
    image: torch.Tensor,
    output_dir: Path,
    model,
    patch_size: int,
    layer: int = -1,
    threshold: float = 0.5,
) -> None:
    img = image.unsqueeze(0)
    h_img, w_img = img.shape[-2:]
    h_featmap, w_featmap = h_img // patch_size, w_img // patch_size
    img = v2.functional.center_crop(img, output_size=[h_featmap * patch_size, w_featmap * patch_size])

    attention_maps = []

    def get_attention_maps(_module, _module_input, module_output):
        attention_maps.append(module_output)

    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
        block.attn.attn_drop.register_forward_hook(get_attention_maps)

    model.eval()
    with torch.no_grad():
        _ = model(img)

    attention_map = attention_maps[layer]  # (batch_size, num_heads, num_patches, num_patches)
    num_heads = attention_map.shape[1]
    cls_attention: torch.Tensor = attention_map[0, :, 0, 1:].squeeze()  # .cpu().numpy()  # (num_heads, num_patches-1)
    cls_attention = cls_attention / torch.sum(cls_attention, dim=-1, keepdim=True)

    if threshold > 0:
        sorted_values, sorted_indices = torch.sort(cls_attention, dim=-1, descending=False, stable=True)
        cumulative_values = torch.cumsum(sorted_values, dim=-1)
        attention_mask_unordered = cumulative_values > (1 - threshold)
        reverse_indices = torch.argsort(sorted_indices)
        attention_mask_ordered = torch.gather(attention_mask_unordered, dim=-1, index=reverse_indices)
        cls_attention = cls_attention * attention_mask_ordered

    attentions = cls_attention.reshape(num_heads, h_featmap, w_featmap)
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0),
        scale_factor=patch_size,
        mode="nearest",
    )[0]

    rows = int(math.sqrt(num_heads))
    cols = math.ceil(num_heads / rows)
    fig, axs = plt.subplots(rows, cols)
    flattened_axs = axs.reshape(-1)
    acc_attention = torch.zeros_like(attentions[0])
    for j in range(num_heads):
        flattened_axs[j].imshow(attentions[j], cmap="viridis", interpolation="nearest")
        flattened_axs[j].axis("off")
        flattened_axs[j].set_title(f"Head {j+1}")
        acc_attention += attentions[j]

    plt.tight_layout()
    plt.savefig(output_dir / "attentions_heads.pdf")
    plt.close(fig)

    fig, axs = plt.subplots()
    axs.imshow(acc_attention / torch.sum(acc_attention))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_segmentation.pdf")


def plot_clusters(model, eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], output_dir: Path):
    if hasattr(eval_loader.dataset, "classes"):
        sums = torch.zeros(len(eval_loader.dataset.classes), model.num_features)  # Initialize a tensor to hold sums
        normalization_factors = torch.zeros(len(eval_loader.dataset.classes))
    else:
        msg = "Ensure that the underlying dataset has an attributed named 'classes'."
        raise ValueError(msg)
    for images, targets in tqdm(eval_loader):
        features = model(images.to(device))
        sums.index_add_(0, index=targets, source=features)
        normalization_factors.index_add_(0, index=targets, source=torch.ones_like(targets, dtype=torch.float))

    normalization_factors[normalization_factors == 0] = 1  # avoid division by zero
    cluster_centroids = (sums / normalization_factors[:, None]).detach().numpy()

    # Calculate PCA embeddings
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cluster_centroids)

    # Create plots
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c="tab:blue", s=50)
    ax.set_title("PCA Class Embeddings")

    synset_names = []
    for synset_id in eval_loader.dataset.classes:
        synset = wordnet.synset_from_pos_and_offset("n", int(synset_id[1:]))
        synset_names.append(synset.lemma_names()[0].replace("_", " "))

    texts = []
    for i, (x, y) in enumerate(pca_result):
        texts.append(plt.text(x, y, synset_names[i], fontsize=8))

    adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5})

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_centroids_pca.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO visualizations.")

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["augmentations", "attention"],
        default="attention",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=Path,
        default=None,
        help="Path to 'deit_small_patch16_224' pretrained weights.",
    )

    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    img = Image.open(args.image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ],
    )
    tensor_image = transform(img)
    if args.type == "augmentations":
        plot_original_image(tensor_image, Path(args.output_dir))
        plot_augmentations(tensor_image, Path(args.output_dir))

    elif args.type == "attention":
        if args.model_path is None:
            raise ValueError("Specify the model path with '-m' or '--model-path.'")

        model: VisionTransformer = timm.create_model("deit_small_patch16_224", num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

        # Assume square patches.
        assert model.patch_embed.patch_size[0] == model.patch_embed.patch_size[1]

        plot_original_image(tensor_image, Path(args.output_dir))
        plot_attention(
            image=tensor_image,
            output_dir=Path(args.output_dir),
            model=model,
            patch_size=model.patch_embed.patch_size[0],
            layer=-1,
            threshold=0.8,
        )


if __name__ == "__main__":
    main()
