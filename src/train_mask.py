import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.defaults import *

from src.dataset import CellDataset
from src.model import ResUNetWithAttentionMask
from src.mask_engine import train_mask_pipeline


# =========================
# Device
# =========================
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# =========================
# Dataset
# =========================
train_dataset = CellDataset(
    os.path.join(ROOT, "train", "images"),
    os.path.join(ROOT, "train", "masks"),
    image_size=IMG_SIZE
)

val_dataset = CellDataset(
    os.path.join(ROOT, "val", "images"),
    os.path.join(ROOT, "val", "masks"),
    image_size=IMG_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# =========================
# Mask Generator
# =========================
model = ResUNetWithAttentionMask(
    in_ch=3,
    out_ch=1
).to(device)


# =========================
# Optimizer
# =========================
optimizer = optim.Adam(
    model.parameters(),
    lr=LR_G,
    betas=BETAS
)


# =========================
# Run
# =========================
if __name__ == "__main__":
    train_mask_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        save_path=MASK_SAVE_PATH,
        device=device,
        num_epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA
    )