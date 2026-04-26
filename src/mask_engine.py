import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs.defaults import *
from src.utils import assert_finite
from src.losses import (
    total_mask_loss,
    dice_coefficient_from_logits,
    iou_from_logits,
    pixel_accuracy_from_logits
)


def init_mask_history():
    keys = ["loss", "dice", "iou", "acc", "bce", "dice_loss", "edge"]

    return {
        "train": {k: [] for k in keys},
        "val": {k: [] for k in keys}
    }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def _is_improved(self, current):
        if self.best is None:
            return True

        if self.mode == "max":
            return current > self.best + self.min_delta

        return current < self.best - self.min_delta

    def step(self, current):
        if self._is_improved(current):
            self.best = float(current)
            self.num_bad = 0
            return False

        self.num_bad += 1
        return self.num_bad >= self.patience


def train_mask_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_edge = 0.0
    n = 0

    for batch in tqdm(loader, desc="Training Mask"):
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        assert_finite(logits, "mask_logits")

        loss, parts = total_mask_loss(
            logits,
            masks,
            lambda_bce=LAMBDA_MASK_BCE,
            lambda_dice=LAMBDA_MASK_DICE,
            lambda_edge=LAMBDA_MASK_EDGE
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = images.size(0)

        dice = dice_coefficient_from_logits(logits, masks)
        iou = iou_from_logits(logits, masks)
        acc = pixel_accuracy_from_logits(logits, masks)

        total_loss += float(loss.detach().item()) * B
        total_dice += float(dice.detach().item()) * B
        total_iou += float(iou.detach().item()) * B
        total_acc += float(acc.detach().item()) * B
        total_bce += float(parts["bce"].detach().item()) * B
        total_dice_loss += float(parts["dice_loss"].detach().item()) * B
        total_edge += float(parts["edge"].detach().item()) * B

        n += B

    return {
        "loss": total_loss / max(1, n),
        "dice": total_dice / max(1, n),
        "iou": total_iou / max(1, n),
        "acc": total_acc / max(1, n),
        "bce": total_bce / max(1, n),
        "dice_loss": total_dice_loss / max(1, n),
        "edge": total_edge / max(1, n),
    }


@torch.no_grad()
def evaluate_mask(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_edge = 0.0
    n = 0

    for batch in tqdm(loader, desc="Validation Mask", leave=False):
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        assert_finite(logits, "val_mask_logits")

        loss, parts = total_mask_loss(
            logits,
            masks,
            lambda_bce=LAMBDA_MASK_BCE,
            lambda_dice=LAMBDA_MASK_DICE,
            lambda_edge=LAMBDA_MASK_EDGE
        )

        B = images.size(0)

        dice = dice_coefficient_from_logits(logits, masks)
        iou = iou_from_logits(logits, masks)
        acc = pixel_accuracy_from_logits(logits, masks)

        total_loss += float(loss.detach().item()) * B
        total_dice += float(dice.detach().item()) * B
        total_iou += float(iou.detach().item()) * B
        total_acc += float(acc.detach().item()) * B
        total_bce += float(parts["bce"].detach().item()) * B
        total_dice_loss += float(parts["dice_loss"].detach().item()) * B
        total_edge += float(parts["edge"].detach().item()) * B

        n += B

    return {
        "loss": total_loss / max(1, n),
        "dice": total_dice / max(1, n),
        "iou": total_iou / max(1, n),
        "acc": total_acc / max(1, n),
        "bce": total_bce / max(1, n),
        "dice_loss": total_dice_loss / max(1, n),
        "edge": total_edge / max(1, n),
    }


def save_best_mask(model, save_path, epoch, val_metrics):
    os.makedirs(save_path, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "tag": "BEST_DICE",
        "metric": "dice",
        "value": float(val_metrics["dice"]),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()}
    }, os.path.join(save_path, "best_DICE.pth"))


def resume_mask_training(model, optimizer, save_path, device):
    ckpt_path = os.path.join(save_path, "checkpoint_latest.pth")

    start_epoch = 0
    history = None
    best_state = None

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("metrics_history", None)
        best_state = ckpt.get("best_state", None)

        print(f"Resumed mask training from epoch {start_epoch}")
    else:
        print("Starting mask training from scratch.")

    if history is None:
        history = init_mask_history()

    if best_state is None:
        best_state = {"best_dice": -1.0}

    return start_epoch, history, best_state


def train_mask_pipeline(
    model,
    train_loader,
    val_loader,
    optimizer,
    save_path,
    device,
    num_epochs=100,
    early_stop_patience=10,
    early_stop_min_delta=0.0
):
    os.makedirs(save_path, exist_ok=True)

    start_epoch, history, best_state = resume_mask_training(
        model, optimizer, save_path, device
    )

    early_stopper = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        mode="max"
    )

    for epoch in range(start_epoch, num_epochs):
        print(f"\n========== Mask Epoch {epoch+1}/{num_epochs} ==========")

        train_m = train_mask_one_epoch(model, train_loader, optimizer, device)
        val_m = evaluate_mask(model, val_loader, device)

        print(
            f"Train | loss={train_m['loss']:.4f} | "
            f"dice={train_m['dice']:.4f} | "
            f"iou={train_m['iou']:.4f} | "
            f"acc={train_m['acc']:.4f}"
        )

        print(
            f"Val   | loss={val_m['loss']:.4f} | "
            f"dice={val_m['dice']:.4f} | "
            f"iou={val_m['iou']:.4f} | "
            f"acc={val_m['acc']:.4f}"
        )

        for k in history["train"].keys():
            history["train"][k].append(train_m[k])

        for k in history["val"].keys():
            history["val"][k].append(val_m[k])

        if val_m["dice"] > best_state["best_dice"]:
            best_state["best_dice"] = float(val_m["dice"])
            save_best_mask(model, save_path, epoch, val_m)
            print(f"Saved BEST_DICE: dice={val_m['dice']:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics_history": history,
            "best_state": best_state,
            "task": "mask_generator_resunet_attention"
        }, os.path.join(save_path, "checkpoint_latest.pth"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stop = early_stopper.step(val_m["dice"])

        print(
            f"EarlyStop monitor=DICE "
            f"value={val_m['dice']:.6f} | "
            f"best={early_stopper.best:.6f} | "
            f"bad_epochs={early_stopper.num_bad}/{early_stopper.patience}"
        )

        if stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("🏁 Mask training finished.")

    return history, best_state