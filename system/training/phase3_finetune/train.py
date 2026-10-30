"""Fase 3: fine-tuning end-to-end (backbone descongelado + RIDIRP + PMA +
sigma_i activo). Se repite una vez por cada uno de los 5 folds, continuando
DESDE el checkpoint de Fase 2 de ese mismo fold (warm-start) + el backbone
de Fase 1 - nunca mezcla folds ni toca el test set fijo.

LR diferenciado (igual criterio que backbone-vs-decoder en Fase 1): el
backbone ya viene afinado dos veces (Fase 1 + warm-start indirecto via
Fase 2), se mueve mas lento; el resto del modelo (ya calibrado por Fase 2,
no al azar) se mueve mas rapido pero menos agresivo que en Fase 2 (ya no
arranca de cero).

EPOCHS=20 (no 30 como Fase 1/2): Fase 2 convergio en 4-14 epocas de 30
(evidencia real, ver su README) y Fase 3 arranca ya calibrada por ese
warm-start - de encima converger mas rapido. El checkpoint de mejor
mean_bal_acc sigue protegiendo contra sobreajuste de todas formas.
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from dataset import FinetuneBagDataset, compute_class_weights, compute_clinical_norm_stats, load_phase3_rows
from losses import Phase3Loss
from model import FinetuneModel

PHASE1_CHECKPOINT = Path(__file__).resolve().parent.parent / "phase1_segmentation" / "checkpoints" / "backbone_best.pt"
PHASE2_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "phase2_warmup" / "checkpoints"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 8
# PyTorch no soporta -1 como "automatico" (a diferencia de sklearn) - se calcula
# segun los nucleos reales de la maquina, con un techo: en Windows, muchos
# procesos worker inicializando CUDA/cuDNN al mismo tiempo generan una
# condicion de carrera real al cargar las DLLs (WinError 1114, verificado -
# paso con 15 workers, nunca con 2-4 en Fase 1/2). Mas workers de los que la
# carga de imagenes en CPU puede aprovechar tampoco ayuda - rendimientos
# decrecientes despues de cierto punto.
NUM_WORKERS = min(max((os.cpu_count() or 1) - 1, 0), 4)
EPOCHS = 20
LR_BACKBONE = 1e-5
LR_REST = 5e-5
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch_train(model, loader, loss_fn, optimizer, scaler, device) -> dict:
    model.train()
    totals = {"total": 0.0, "stage": 0.0, "plus": 0.0, "ap_rop": 0.0}
    n_batches = 0
    for clinical, images, bag_mask, targets, valid in loader:
        clinical, images, bag_mask = clinical.to(device), images.to(device), bag_mask.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        valid = {k: v.to(device) for k, v in valid.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(clinical, images, bag_mask)
            total_loss, parts = loss_fn(outputs, targets, valid)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += total_loss.item()
        for k in ("stage", "plus", "ap_rop"):
            totals[k] += parts[k]
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def run_epoch_val(model, loader, loss_fn, device) -> tuple[dict, dict, dict]:
    model.eval()
    totals = {"total": 0.0, "stage": 0.0, "plus": 0.0, "ap_rop": 0.0}
    n_batches = 0
    preds = {"stage": [], "plus": [], "ap_rop": []}
    trues = {"stage": [], "plus": [], "ap_rop": []}
    last_sigma = {}

    for clinical, images, bag_mask, targets, valid in loader:
        clinical, images, bag_mask = clinical.to(device), images.to(device), bag_mask.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        valid = {k: v.to(device) for k, v in valid.items()}

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(clinical, images, bag_mask)
            total_loss, parts = loss_fn(outputs, targets, valid)
        totals["total"] += total_loss.item()
        for k in ("stage", "plus", "ap_rop"):
            totals[k] += parts[k]
        last_sigma = parts["sigma"]
        n_batches += 1

        stage_pred = outputs["stage_logits"].argmax(dim=1)
        plus_pred = outputs["plus_logits"].argmax(dim=1)
        ap_rop_pred = (torch.sigmoid(outputs["ap_rop_logits"]) >= 0.5).long()

        for name, pred, true, m in [
            ("stage", stage_pred, targets["stage"], valid["stage"]),
            ("plus", plus_pred, targets["plus"], valid["plus"]),
            ("ap_rop", ap_rop_pred, targets["ap_rop"].long(), valid["ap_rop"]),
        ]:
            if m.any():
                preds[name].extend(pred[m].cpu().tolist())
                trues[name].extend(true[m].cpu().tolist())

    bal_acc = {}
    for name in ["stage", "plus", "ap_rop"]:
        if len(trues[name]) > 0 and len(set(trues[name])) > 1:
            bal_acc[name] = balanced_accuracy_score(trues[name], preds[name])
        else:
            bal_acc[name] = float("nan")

    avg_loss = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return avg_loss, bal_acc, last_sigma


def run_fold(fold: int, device: torch.device) -> float:
    set_seed(SEED + fold)
    df = load_phase3_rows()
    train_df = df[(df["split"] == "train") & (df["fold"] != fold)]
    val_df = df[(df["split"] == "train") & (df["fold"] == fold)]

    norm_stats = compute_clinical_norm_stats(train_df)
    class_weights = compute_class_weights(train_df)

    phase1_ckpt = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
    phase2_ckpt_path = PHASE2_CHECKPOINT_DIR / f"fold{fold}_best.pt"
    phase2_ckpt = torch.load(phase2_ckpt_path, map_location=device, weights_only=False)
    data_cfg_mean, data_cfg_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # mismo checkpoint que Fase 1/2, mean=std=0.5

    train_ds = FinetuneBagDataset(train_df, norm_stats, data_cfg_mean, data_cfg_std, train=True, seed=SEED + fold)
    val_ds = FinetuneBagDataset(val_df, norm_stats, data_cfg_mean, data_cfg_std, train=False, seed=SEED + fold)
    print(f"[fold {fold}] train bags: {len(train_ds)} | val bags: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                               persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None)

    model = FinetuneModel(
        backbone_name=phase1_ckpt["backbone_name"],
        pretrained_backbone_state_dict=phase1_ckpt["backbone_state_dict"],
    ).to(device)
    model.load_warmup_weights(phase2_ckpt["model_state_dict"])

    loss_fn = Phase3Loss(
        stage_alpha=class_weights["stage_alpha"],
        plus_weights=class_weights["plus_weights"],
        ap_rop_pos_weight=class_weights["ap_rop_pos_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")], "lr": LR_REST},
            {"params": loss_fn.log_var, "lr": LR_REST},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_mean_bal_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_losses = run_epoch_train(model, train_loader, loss_fn, optimizer, scaler, device)
        val_losses, bal_acc, sigma = run_epoch_val(model, val_loader, loss_fn, device)
        scheduler.step()

        valid_scores = [v for v in bal_acc.values() if not np.isnan(v)]
        mean_bal_acc = float(np.mean(valid_scores)) if valid_scores else 0.0

        acc_str = ", ".join(f"{k}={v:.3f}" if not np.isnan(v) else f"{k}=n/a" for k, v in bal_acc.items())
        sigma_str = ", ".join(f"{k}={v:.3f}" for k, v in sigma.items())
        print(f"[fold {fold}][epoca {epoch:02d}/{EPOCHS}] train_loss={train_losses['total']:.4f} "
              f"val_loss={val_losses['total']:.4f} bal_acc: {acc_str} | mean={mean_bal_acc:.4f} | sigma: {sigma_str}")

        if mean_bal_acc > best_mean_bal_acc:
            best_mean_bal_acc = mean_bal_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_name": phase1_ckpt["backbone_name"],
                    "norm_stats": norm_stats,
                    "class_weights": class_weights,
                    "fold": fold,
                    "epoch": epoch,
                    "mean_bal_acc": mean_bal_acc,
                    "bal_acc_per_task": bal_acc,
                    "sigma_per_task": sigma,
                },
                CHECKPOINT_DIR / f"fold{fold}_best.pt",
            )

    print(f"[fold {fold}] mejor mean_bal_acc: {best_mean_bal_acc:.4f}")
    return best_mean_bal_acc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    results = []
    for fold in range(N_FOLDS):
        existing = CHECKPOINT_DIR / f"fold{fold}_best.pt"
        if existing.exists():
            # Reanuda folds ya completados (ej. tras una desconexion de Colab) -
            # nunca reentrena un fold que ya tiene checkpoint guardado.
            ckpt = torch.load(existing, map_location="cpu", weights_only=False)
            print(f"[fold {fold}] checkpoint ya existe (mean_bal_acc={ckpt['mean_bal_acc']:.4f}), se omite")
            results.append(ckpt["mean_bal_acc"])
            continue
        results.append(run_fold(fold, device))

    print("\n[resumen] mean_bal_acc por fold:", [round(r, 4) for r in results])
    print(f"[resumen] promedio entre folds: {np.mean(results):.4f} +- {np.std(results):.4f}")


if __name__ == "__main__":
    main()
