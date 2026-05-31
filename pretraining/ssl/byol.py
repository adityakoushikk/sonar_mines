"""BYOL self-supervised pretraining of a YOLO backbone on SSS tiles (B6).

Online network predicts a momentum network's projection of a second view (no
negatives). After training we serialize only the backbone via the checkpoint
contract in ``backbone.py``. ``METHOD_REGISTRY`` leaves room for a future DINO.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch import Tensor, nn

from pretraining.ssl.backbone import extract_yolo_backbone, save_backbone_checkpoint
from pretraining.ssl.data import build_webdataset_loader
from pretraining.ssl.transforms import SonarBYOLTransform


def _final_ckpt_name(variant: str) -> str:
    """Variant-scoped checkpoint name so v8/v26 runs don't overwrite each other."""
    return f"byol_benthicat_{variant}.pt"


@dataclass
class TrainConfig:
    """Hyperparameters and run settings for self-supervised pretraining."""

    shards: str
    out_dir: str
    variant: str = "yolov8n"
    epochs: int = 100
    batch_size: int = 256
    num_workers: int = 8
    lr: float = 1e-3
    momentum_base: float = 0.996
    ckpt_every_epochs: int = 10
    wandb_enabled: bool = False
    wandb_project: str = "sonar-ssl"
    epoch_length: int | None = None
    mixed_precision: str = "bf16"  # forced to "no" off-CUDA


class BYOL(nn.Module):
    """BYOL online+target wrapper around a YOLO backbone (pool -> project -> predict)."""

    def __init__(self, backbone: nn.Sequential, in_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = BYOLProjectionHead(in_dim, 4096, 256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)

        # Target network: frozen deepcopies, EMA-tracked by the trainer.
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _embed(self, backbone: nn.Module, x: Tensor) -> Tensor:
        return self.pool(backbone(x)).flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.prediction_head(self.projection_head(self._embed(self.backbone, x)))

    def forward_momentum(self, x: Tensor) -> Tensor:
        y = self._embed(self.backbone_momentum, x)
        return self.projection_head_momentum(y).detach()


def _build_byol(cfg: TrainConfig) -> tuple[object, BYOL, int]:
    yolo, backbone, channels = extract_yolo_backbone(variant=cfg.variant)
    return yolo, BYOL(backbone, in_dim=channels), channels


METHOD_REGISTRY: dict[str, Callable[[TrainConfig], tuple[object, BYOL, int]]] = {
    "byol": _build_byol,
}


def train_byol(
    cfg: TrainConfig,
    method: str = "byol",
    on_checkpoint: Callable[[str], None] | None = None,
) -> str:
    """Run SSL pretraining via ``accelerate`` and return the final checkpoint path.

    Loss is symmetric negative-cosine over the two views; the target net is
    EMA-updated each step. Runs single-process on CPU (smoke) or N-GPU via DDP.
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"unknown SSL method {method!r}; registered: {sorted(METHOD_REGISTRY)}"
        )

    cuda = torch.cuda.is_available()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # dispatch_batches=False so each rank reads its OWN shard split (split_by_node);
    # accelerate's IterableDataset default of True would drop all but rank 0's data.
    dl_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision if cuda else "no",
        dataloader_config=dl_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Multi-GPU needs a fixed epoch_length so every rank yields equal batches,
    # else DDP hangs at the epoch boundary on an unbounded WebDataset.
    if accelerator.num_processes > 1 and cfg.epoch_length is None:
        raise ValueError(
            "multi-GPU runs require cfg.epoch_length (set it to ~tiles_per_rank) "
            "so every rank yields equal batches and DDP doesn't hang at epoch end"
        )

    use_wandb = cfg.wandb_enabled and accelerator.is_main_process
    if use_wandb:
        import wandb

        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    yolo, model, channels = METHOD_REGISTRY[method](cfg)

    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    transform = SonarBYOLTransform()
    loader = build_webdataset_loader(
        cfg.shards,
        transform=transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        world_size=accelerator.num_processes,
        epoch_length=cfg.epoch_length,
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    out_dir = Path(cfg.out_dir)
    final_path = out_dir / _final_ckpt_name(cfg.variant)

    model.train()
    for epoch in range(cfg.epochs):
        # update_momentum / save touch raw attributes, so use the unwrapped model.
        unwrapped = accelerator.unwrap_model(model)

        epoch_loss = 0.0
        n_batches = 0
        for view0, view1 in loader:
            p0 = model(view0)
            p1 = model(view1)
            z0 = unwrapped.forward_momentum(view0)
            z1 = unwrapped.forward_momentum(view1)

            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            update_momentum(
                unwrapped.backbone, unwrapped.backbone_momentum, m=cfg.momentum_base
            )
            update_momentum(
                unwrapped.projection_head,
                unwrapped.projection_head_momentum,
                m=cfg.momentum_base,
            )

            epoch_loss += float(loss.detach().item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if accelerator.is_main_process:
            print(f"[byol] epoch {epoch} loss {avg_loss:.4f} ({n_batches} batches)")
            if use_wandb:
                import wandb

                wandb.log({"train/loss": avg_loss, "epoch": epoch})

        is_periodic = (epoch + 1) % max(1, cfg.ckpt_every_epochs) == 0
        if accelerator.is_main_process and is_periodic:
            save_backbone_checkpoint(
                yolo, final_path, channels=channels, epoch=epoch, variant=cfg.variant
            )
            if on_checkpoint is not None:
                on_checkpoint(str(final_path))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_backbone_checkpoint(
            yolo, final_path, channels=channels, epoch=cfg.epochs - 1, variant=cfg.variant
        )
        if on_checkpoint is not None:
            on_checkpoint(str(final_path))
        if use_wandb:
            import wandb

            wandb.finish()

    return str(final_path)
