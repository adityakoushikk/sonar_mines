"""BYOL self-supervised pretraining of a YOLOv8 backbone on SSS tiles (B6).

BYOL ("Bootstrap Your Own Latent") trains an online network to predict the
projection of a momentum (target) network on a second augmented view, with no
negative pairs. We attach lightly's BYOL projection/prediction heads on top of
a globally-pooled YOLOv8 backbone, and after training serialize *only* the
backbone via the shared CHECKPOINT CONTRACT (see ``backbone.py``) so the
supervised loader can drop it into a fresh detector.

A small ``METHOD_REGISTRY`` maps a method name to a ``(model_builder)`` so a
future DINO variant can slot in behind the same ``train_byol``-style interface
without touching the training loop's structure. Only BYOL is implemented here.
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
    """Variant-scoped final checkpoint filename, written under ``cfg.out_dir``.

    Encoding the variant keeps a yolov8n and a yolo26n run from overwriting each
    other's checkpoint. Matches the path baked into the per-variant init configs'
    ``weights_path`` (e.g. ``byol_benthicat_yolov8n.pt``).
    """
    return f"byol_benthicat_{variant}.pt"


@dataclass
class TrainConfig:
    """Hyperparameters and run settings for self-supervised pretraining.

    Attributes:
        shards: WebDataset shard spec (brace-glob string or list of paths).
        out_dir: Directory for checkpoints (final + periodic).
        variant: YOLO architecture variant, e.g. ``"yolov8n"``.
        epochs: Number of training epochs.
        batch_size: Per-step batch size (number of tiles; each yields two views).
        num_workers: DataLoader workers.
        lr: AdamW learning rate.
        momentum_base: Base EMA momentum for the target network update.
        ckpt_every_epochs: Write a periodic checkpoint every N epochs.
        wandb_enabled: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        epoch_length: If set, fixes samples/epoch via WebDataset ``.with_epoch``.
        mixed_precision: Accelerate precision on CUDA (``"bf16"``/``"fp16"``/``"no"``);
            forced to ``"no"`` when CUDA is unavailable.
    """

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
    mixed_precision: str = "bf16"


class BYOL(nn.Module):
    """BYOL online+target wrapper around a YOLOv8 backbone.

    The online path is ``backbone -> global avg pool -> projection -> prediction``.
    The target (momentum) path is a deepcopy of the backbone + projection with
    gradients deactivated; its weights are EMA-updated from the online weights
    each step by the trainer (via :func:`lightly.models.utils.update_momentum`).

    Args:
        backbone: The ``nn.Sequential`` backbone from
            :func:`extract_yolo_backbone`. Trained in place (shares tensors with
            the parent YOLO so the checkpoint sees the updates).
        in_dim: Channel count of the backbone's output feature map (the SPPF
            channels inferred by ``extract_yolo_backbone``).
    """

    def __init__(self, backbone: nn.Sequential, in_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        # Collapse the HxW feature map to a per-channel vector so the MLP heads
        # operate on a fixed (B, in_dim) tensor regardless of input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # lightly's standard BYOL head shapes: project to 256-d via a 4096-d
        # hidden layer, predict in the same space.
        self.projection_head = BYOLProjectionHead(in_dim, 4096, 256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)

        # Target network: frozen deepcopies, EMA-tracked by the trainer.
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _embed(self, backbone: nn.Module, x: Tensor) -> Tensor:
        """Run ``x`` through ``backbone`` then global-pool to ``(B, in_dim)``."""
        y = backbone(x)
        return self.pool(y).flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Online path: return the BYOL *prediction* ``p`` for view ``x``."""
        y = self._embed(self.backbone, x)
        z = self.projection_head(y)
        return self.prediction_head(z)

    def forward_momentum(self, x: Tensor) -> Tensor:
        """Target path: return the detached momentum *projection* ``z`` for ``x``."""
        y = self._embed(self.backbone_momentum, x)
        # Detach: no gradient flows into the target network (it is EMA-updated).
        return self.projection_head_momentum(y).detach()


def _build_byol(cfg: TrainConfig) -> tuple[object, BYOL, int]:
    """Construct (yolo, BYOL module, channels) for a BYOL run.

    Factored out behind :data:`METHOD_REGISTRY` so an alternative SSL method can
    provide its own builder with the same return shape later.
    """
    yolo, backbone, channels = extract_yolo_backbone(variant=cfg.variant)
    model = BYOL(backbone, in_dim=channels)
    return yolo, model, channels


# Method registry: name -> builder. DINO etc. can register here and reuse the
# train loop's plumbing (loader, accelerate, checkpointing) without forking it.
METHOD_REGISTRY: dict[str, Callable[[TrainConfig], tuple[object, BYOL, int]]] = {
    "byol": _build_byol,
}


def train_byol(
    cfg: TrainConfig,
    method: str = "byol",
    on_checkpoint: Callable[[str], None] | None = None,
) -> str:
    """Run self-supervised pretraining and return the final checkpoint path.

    Uses ``accelerate.Accelerator`` for device placement and mixed precision.
    Mixed precision is only enabled on CUDA; on CPU (the smoke-test path) it
    degrades to ``"no"`` and every CUDA-only call is guarded, so a single CPU
    process trains end-to-end.

    The BYOL loss is symmetric negative-cosine-similarity over the two views:
    ``0.5 * (crit(p0, z1) + crit(p1, z0))``. After each optimizer step the target
    network is EMA-updated from the online network with momentum ``cfg.momentum_base``.

    DDP note: the momentum sub-networks have ``requires_grad=False``. Under DDP
    that trips "parameters that didn't receive grad" errors, so we pass
    ``DistributedDataParallelKwargs(find_unused_parameters=True)`` and run
    ``update_momentum`` on the *unwrapped* model.

    Args:
        cfg: Run configuration.
        method: Key into :data:`METHOD_REGISTRY` (only ``"byol"`` implemented).

    Returns:
        Absolute path (as a string) to the variant-scoped checkpoint under
        ``cfg.out_dir`` (e.g. ``byol_benthicat_yolov8n.pt``).
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"unknown SSL method {method!r}; registered: {sorted(METHOD_REGISTRY)}"
        )

    cuda = torch.cuda.is_available()
    # find_unused_parameters=True keeps DDP happy about the frozen momentum nets.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # dispatch_batches=False: each rank must iterate its OWN WebDataset (we shard
    # with split_by_node). Accelerate defaults this to True for IterableDatasets,
    # which would make only rank 0 read data — and our split_by_node would then
    # see just 1/world_size of the shards: silent data loss on multi-GPU.
    dl_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision if cuda else "no",
        dataloader_config=dl_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Multi-GPU + an unbounded WebDataset can hand ranks unequal batch counts,
    # which deadlocks DDP at the epoch boundary (one rank stops all-reducing while
    # the others wait on it). A fixed epoch_length makes every rank yield the same
    # number of batches. Single-process runs are unaffected.
    if accelerator.num_processes > 1 and cfg.epoch_length is None:
        raise ValueError(
            "multi-GPU runs require cfg.epoch_length (set it to ~tiles_per_rank) "
            "so every rank yields equal batches and DDP doesn't hang at epoch end"
        )

    # W&B only on the main process, only when asked.
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

    # Accelerate moves model+optimizer to device and wraps in DDP if distributed.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    out_dir = Path(cfg.out_dir)
    final_path = out_dir / _final_ckpt_name(cfg.variant)

    model.train()
    for epoch in range(cfg.epochs):
        # update_momentum / save touch the raw module attributes, so resolve the
        # unwrapped model once per epoch (it is a no-op wrapper-strip on CPU).
        unwrapped = accelerator.unwrap_model(model)

        epoch_loss = 0.0
        n_batches = 0
        for view0, view1 in loader:
            # Online predictions for both views.
            p0 = model(view0)
            p1 = model(view1)
            # Target projections for both views (detached inside forward_momentum).
            z0 = unwrapped.forward_momentum(view0)
            z1 = unwrapped.forward_momentum(view1)

            # Symmetric BYOL loss: predict each view's target from the other view.
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # EMA-update the target network from the online network. Run on the
            # unwrapped model so DDP's wrapper does not interfere with the copy.
            update_momentum(unwrapped.backbone, unwrapped.backbone_momentum, m=cfg.momentum_base)
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

        # Periodic checkpoint (main process only). epoch+1 so epoch 9 with
        # ckpt_every=10 -> writes at the 10th epoch boundary.
        # max(1, ...) guards against a misconfigured ckpt_every_epochs == 0.
        is_periodic = (epoch + 1) % max(1, cfg.ckpt_every_epochs) == 0
        if accelerator.is_main_process and is_periodic:
            save_backbone_checkpoint(
                yolo, final_path, channels=channels, epoch=epoch, variant=cfg.variant
            )
            if on_checkpoint is not None:
                on_checkpoint(str(final_path))

    # Final checkpoint (always, main process). The backbone shares tensors with
    # `yolo`, so its trained weights are already reflected in yolo.model.
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
