"""Fase 2: warm-up multimodal (visual congelado) - entrena MLP clinico +
Cross-Attention + Attention Pooling + 3 cabezales sobre ROP-VL + FARFUM.

Se repite UNA VEZ POR CADA UNO DE LOS 5 FOLDS del Group K-Fold clinico
(README principal S2: "solo Fase 2/3 se repite K veces" - a diferencia de
Fase 1, cuyo backbone es unico y se reutiliza sin cambios en los 5 folds).
Cada fold entrena modulos nuevos desde cero, usando SOLO su propio train
(los otros 4 folds) para calcular normalizacion/pesos de clase y
entrenar, y valida contra su propio fold reservado - nunca toca el test
set fijo ni mezcla folds, para no filtrar informacion entre ellos (el
checkpoint de cada fold es el punto de partida de la Fase 3 de ESE mismo
fold, no se comparte entre folds).

Perdida: suma simple de las 3 tareas enmascaradas, SIN pesos de
incertidumbre sigma_i (ver losses.py y la discusion de diseño - sigma_i se
activa recien en Fase 3).

Metrica de validacion: balanced accuracy por cabezal (promedio de recall
por clase - mas robusta que accuracy simple dado el desbalance real de
plus/ap_rop), calculada sobre todo el val set del fold, no por batch.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from dataset import WarmupBagDataset, compute_class_weights, compute_clinical_norm_stats, load_phase2_rows
from losses import Phase2Loss
from model import WarmupModel

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch_train(model, loader, loss_fn, optimizer, scaler, device) -> dict:
    model.train()
    totals = {"total": 0.0, "stage": 0.0, "plus": 0.0, "ap_rop": 0.0}
    n_batches = 0
    for clinical, visual, bag_mask, targets, valid in loader:
        clinical, visual, bag_mask = clinical.to(device), visual.to(device), bag_mask.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        valid = {k: v.to(device) for k, v in valid.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(clinical, visual, bag_mask)
            total_loss, parts = loss_fn(outputs, targets, valid)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += total_loss.item()
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def run_epoch_val(model, loader, loss_fn, device) -> tuple[dict, dict]:
    model.eval()
    totals = {"total": 0.0, "stage": 0.0, "plus": 0.0, "ap_rop": 0.0}
    n_batches = 0
    preds = {"stage": [], "plus": [], "ap_rop": []}
    trues = {"stage": [], "plus": [], "ap_rop": []}

    for clinical, visual, bag_mask, targets, valid in loader:
        clinical, visual, bag_mask = clinical.to(device), visual.to(device), bag_mask.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        valid = {k: v.to(device) for k, v in valid.items()}

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(clinical, visual, bag_mask)
            total_loss, parts = loss_fn(outputs, targets, valid)
        totals["total"] += total_loss.item()
        for k, v in parts.items():
            totals[k] += v
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
            bal_acc[name] = float("nan")  # no hay suficiente variedad de clases en el val set para medir

    avg_loss = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return avg_loss, bal_acc


def run_fold(fold: int, device: torch.device) -> float:
    set_seed(SEED + fold)
    df = load_phase2_rows()
    train_df = df[(df["split"] == "train") & (df["fold"] != fold)]
    val_df = df[(df["split"] == "train") & (df["fold"] == fold)]

    norm_stats = compute_clinical_norm_stats(train_df)
    class_weights = compute_class_weights(train_df)

    train_ds = WarmupBagDataset(train_df, norm_stats, train=True, seed=SEED + fold)
    val_ds = WarmupBagDataset(val_df, norm_stats, train=False, seed=SEED + fold)
    print(f"[fold {fold}] train bags: {len(train_ds)} | val bags: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = WarmupModel().to(device)
    loss_fn = Phase2Loss(
        stage_alpha=class_weights["stage_alpha"],
        plus_weights=class_weights["plus_weights"],
        ap_rop_pos_weight=class_weights["ap_rop_pos_weight"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_mean_bal_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_losses = run_epoch_train(model, train_loader, loss_fn, optimizer, scaler, device)
        val_losses, bal_acc = run_epoch_val(model, val_loader, loss_fn, device)
        scheduler.step()

        valid_scores = [v for v in bal_acc.values() if not np.isnan(v)]
        mean_bal_acc = float(np.mean(valid_scores)) if valid_scores else 0.0

        acc_str = ", ".join(f"{k}={v:.3f}" if not np.isnan(v) else f"{k}=n/a" for k, v in bal_acc.items())
        print(f"[fold {fold}][epoca {epoch:02d}/{EPOCHS}] train_loss={train_losses['total']:.4f} "
              f"val_loss={val_losses['total']:.4f} bal_acc: {acc_str} | mean={mean_bal_acc:.4f}")

        if mean_bal_acc > best_mean_bal_acc:
            best_mean_bal_acc = mean_bal_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "norm_stats": norm_stats,
                    "class_weights": class_weights,
                    "fold": fold,
                    "epoch": epoch,
                    "mean_bal_acc": mean_bal_acc,
                    "bal_acc_per_task": bal_acc,
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
        results.append(run_fold(fold, device))

    print("\n[resumen] mean_bal_acc por fold:", [round(r, 4) for r in results])
    print(f"[resumen] promedio entre folds: {np.mean(results):.4f} +- {np.std(results):.4f}")


if __name__ == "__main__":
    main()
