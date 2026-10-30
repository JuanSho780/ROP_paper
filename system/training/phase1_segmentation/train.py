"""Fase 1: fine-tuning del backbone ViT en segmentacion (HVDROPDB).

Entrena ViTSegmentationModel (backbone ViT + decoder multi-escala) con
MaskedDiceBCELoss sobre el split train/val ya calculado por
system/dataset_build/partition/split_hvdropdb.py (agrupado por image_hash,
ver ese script).

Revision 2 (tras la primera corrida de 30 epocas con decoder de una sola
capa y Dice puro, donde BV quedo con Dice ~0.29 pese a tener MAS pixeles
positivos que OD, que llego a 0.57 - evidencia de perdida de detalle fino,
no de desbalance de clases): decoder multi-escala (model.py) + perdida
Dice+BCE (losses.py) + mas epocas. Ver README de esta carpeta.

Validacion: split UNICO train/val, no K-fold - decision ya documentada
(README principal S2/S5): Fase 1 es una etapa auxiliar (el backbone
resultante se congela y se reutiliza igual en los 5 folds de Fase 2/3), no
el resultado que se reporta en la tesis, asi que no se justifica el costo
de cross-validation completa aqui.

Al final guarda:
  - el backbone (solo `model.backbone`, sin el decoder - el decoder se
    descarta, nunca se vuelve a usar despues de Fase 1) en
    system/training/phase1_segmentation/checkpoints/backbone_best.pt
  - un log por epoca (loss de train, Dice de val por estructura y promedio)

Requiere: torch, torchvision, timm, albumentations, opencv-python,
pandas, pyarrow. Ejecutar despues de split_hvdropdb.py.
"""
import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from dataset import STRUCTURES, HVDROPDBSegmentationDataset
from losses import MaskedDiceBCELoss, masked_dice_score
from model import BACKBONE_NAME, ViTSegmentationModel

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # system/training/phase1_segmentation/ -> ROP_paper/
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 60
LR_BACKBONE = 1e-5
LR_DECODER = 1e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(mean: tuple, std: tuple) -> tuple[DataLoader, DataLoader]:
    train_ds = HVDROPDBSegmentationDataset("train", mean=mean, std=std)
    val_ds = HVDROPDBSegmentationDataset("val", mean=mean, std=std)
    print(f"[data] train: {len(train_ds)} imagenes | val: {len(val_ds)} imagenes")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch_train(model, loader, loss_fn, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, target, channel_valid in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        channel_valid = channel_valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = loss_fn(logits, target, channel_valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, loader, loss_fn, device) -> tuple[float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    dice_sum = torch.zeros(len(STRUCTURES), device=device)
    dice_n = torch.zeros(len(STRUCTURES), device=device)

    for images, target, channel_valid in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        channel_valid = channel_valid.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = loss_fn(logits, target, channel_valid)

        total_loss += loss.item()
        n_batches += 1

        dice_batch = masked_dice_score(logits, target, channel_valid)  # (C,)
        valid_mask = channel_valid.any(dim=0).float()  # aporta si al menos 1 muestra del batch tiene ese canal
        dice_sum += dice_batch * valid_mask
        dice_n += valid_mask

    dice_per_structure = dice_sum / dice_n.clamp(min=1)
    return total_loss / max(n_batches, 1), dice_per_structure.cpu()


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data_cfg = timm.data.resolve_data_config({}, model=timm.create_model(BACKBONE_NAME, pretrained=False))
    mean, std = data_cfg["mean"], data_cfg["std"]
    print(f"[normalizacion] mean={mean} std={std} (del checkpoint preentrenado, no ImageNet generico)")

    train_loader, val_loader = build_dataloaders(mean, std)

    model = ViTSegmentationModel().to(device)
    loss_fn = MaskedDiceBCELoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.decoder.parameters(), "lr": LR_DECODER},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_mean_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_train(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss, dice_per_structure = run_epoch_val(model, val_loader, loss_fn, device)
        mean_dice = dice_per_structure.mean().item()
        scheduler.step()

        dice_str = ", ".join(f"{s}={d:.3f}" for s, d in zip(STRUCTURES, dice_per_structure.tolist()))
        print(f"[epoca {epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"dice: {dice_str} | mean_dice={mean_dice:.4f}")

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(
                {
                    "backbone_state_dict": model.backbone.state_dict(),
                    "backbone_name": BACKBONE_NAME,
                    "epoch": epoch,
                    "mean_dice": mean_dice,
                    "dice_per_structure": dict(zip(STRUCTURES, dice_per_structure.tolist())),
                },
                CHECKPOINT_DIR / "backbone_best.pt",
            )
            print(f"  -> nuevo mejor backbone guardado (mean_dice={mean_dice:.4f})")

    print(f"\n[fin] mejor mean_dice de validacion: {best_mean_dice:.4f}")
    print(f"-> {CHECKPOINT_DIR / 'backbone_best.pt'}")


if __name__ == "__main__":
    main()
