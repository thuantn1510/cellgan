import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from configs.defaults import *
from src.dataset import CellDataset


# ============================================================
# Model: U-Net 2D
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_n_filter=64):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_n_filter)
        self.enc2 = DownBlock(base_n_filter, base_n_filter * 2)
        self.enc3 = DownBlock(base_n_filter * 2, base_n_filter * 4)
        self.enc4 = DownBlock(base_n_filter * 4, base_n_filter * 8)
        self.enc5 = DownBlock(base_n_filter * 8, base_n_filter * 16)

        self.up4 = UpBlock(base_n_filter * 16, base_n_filter * 8, base_n_filter * 8)
        self.up3 = UpBlock(base_n_filter * 8, base_n_filter * 4, base_n_filter * 4)
        self.up2 = UpBlock(base_n_filter * 4, base_n_filter * 2, base_n_filter * 2)
        self.up1 = UpBlock(base_n_filter * 2, base_n_filter, base_n_filter)

        self.final_conv = nn.Conv2d(base_n_filter, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.enc5(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.final_conv(d1)


# ============================================================
# Model: VNet2D
# ============================================================
def conv_block(in_channels, out_channels, kernel_size=3, padding=1, num_convs=2, activation=nn.PReLU):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        activation(num_parameters=out_channels)
    ]

    for _ in range(num_convs - 1):
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(num_parameters=out_channels)
        ])

    return nn.Sequential(*layers)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, activation=nn.PReLU):
        super().__init__()

        self.conv_layers = conv_block(
            in_channels,
            out_channels,
            num_convs=num_convs,
            activation=activation
        )

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )

        self.activation_final = activation(num_parameters=out_channels)

    def forward(self, x):
        residual = x

        out = self.conv_layers(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        out = out + residual
        out = self.activation_final(out)

        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.PReLU):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(num_parameters=out_channels)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.PReLU):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(num_parameters=out_channels)

    def forward(self, x):
        return self.act(self.bn(self.upconv(x)))


class VNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_n_filter=16, activation=nn.PReLU):
        super().__init__()

        self.enc_block1 = ResidualConvBlock(
            in_channels,
            base_n_filter,
            num_convs=1,
            activation=activation
        )
        self.down1 = DownsampleBlock(
            base_n_filter,
            base_n_filter * 2,
            activation=activation
        )

        self.enc_block2 = ResidualConvBlock(
            base_n_filter * 2,
            base_n_filter * 2,
            num_convs=2,
            activation=activation
        )
        self.down2 = DownsampleBlock(
            base_n_filter * 2,
            base_n_filter * 4,
            activation=activation
        )

        self.enc_block3 = ResidualConvBlock(
            base_n_filter * 4,
            base_n_filter * 4,
            num_convs=3,
            activation=activation
        )
        self.down3 = DownsampleBlock(
            base_n_filter * 4,
            base_n_filter * 8,
            activation=activation
        )

        self.enc_block4 = ResidualConvBlock(
            base_n_filter * 8,
            base_n_filter * 8,
            num_convs=3,
            activation=activation
        )
        self.down4 = DownsampleBlock(
            base_n_filter * 8,
            base_n_filter * 16,
            activation=activation
        )

        self.bottleneck = ResidualConvBlock(
            base_n_filter * 16,
            base_n_filter * 16,
            num_convs=3,
            activation=activation
        )

        self.up4 = UpsampleBlock(
            base_n_filter * 16,
            base_n_filter * 8,
            activation=activation
        )
        self.dec_block4 = ResidualConvBlock(
            base_n_filter * 8 + base_n_filter * 8,
            base_n_filter * 8,
            num_convs=3,
            activation=activation
        )

        self.up3 = UpsampleBlock(
            base_n_filter * 8,
            base_n_filter * 4,
            activation=activation
        )
        self.dec_block3 = ResidualConvBlock(
            base_n_filter * 4 + base_n_filter * 4,
            base_n_filter * 4,
            num_convs=3,
            activation=activation
        )

        self.up2 = UpsampleBlock(
            base_n_filter * 4,
            base_n_filter * 2,
            activation=activation
        )
        self.dec_block2 = ResidualConvBlock(
            base_n_filter * 2 + base_n_filter * 2,
            base_n_filter * 2,
            num_convs=2,
            activation=activation
        )

        self.up1 = UpsampleBlock(
            base_n_filter * 2,
            base_n_filter,
            activation=activation
        )
        self.dec_block1 = ResidualConvBlock(
            base_n_filter + base_n_filter,
            base_n_filter,
            num_convs=1,
            activation=activation
        )

        self.final_conv = nn.Conv2d(
            base_n_filter,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        e1 = self.enc_block1(x)
        d1 = self.down1(e1)

        e2 = self.enc_block2(d1)
        d2 = self.down2(e2)

        e3 = self.enc_block3(d2)
        d3 = self.down3(e3)

        e4 = self.enc_block4(d3)
        d4 = self.down4(e4)

        b = self.bottleneck(d4)

        u4 = self.up4(b)
        if u4.shape[2:] != e4.shape[2:]:
            u4 = F.interpolate(
                u4,
                size=e4.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        dec4 = self.dec_block4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(dec4)
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(
                u3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        dec3 = self.dec_block3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(dec3)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(
                u2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        dec2 = self.dec_block2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(dec2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(
                u1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        dec1 = self.dec_block1(torch.cat([u1, e1], dim=1))

        logits = self.final_conv(dec1)

        return logits
    
# ============================================================
# Model: R50-U-Net
# ============================================================
    

import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class R50UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # fix input channel nếu != 3
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

            if in_channels == 1:
                with torch.no_grad():
                    backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.input_block = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )  # /2

        self.maxpool = backbone.maxpool  # /4

        self.encoder1 = backbone.layer1  # 256
        self.encoder2 = backbone.layer2  # 512
        self.encoder3 = backbone.layer3  # 1024
        self.encoder4 = backbone.layer4  # 2048

        self.center = nn.Sequential(
            ConvBNReLU(2048, 1024),
            ConvBNReLU(1024, 1024)
        )

        self.dec4 = DecoderBlock(1024, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.final_up = nn.Sequential(
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 32)
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.input_block(x)
        x1 = self.maxpool(x0)

        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        c = self.center(x4)

        d4 = self.dec4(c, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.final_up(out)

        return self.final_conv(out)
    
import torchvision.models as models


class DeepLabV3Binary(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True):
        super().__init__()

        self.model = models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        )

        # đổi classifier → 1 channel
        in_ch = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        out = self.model(x)["out"]  # IMPORTANT
        return out

# ============================================================
# Metrics
# ============================================================
def dice_iou_acc_from_logits(logits, target, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    pred_f = pred.view(pred.size(0), -1)
    target_f = target.view(target.size(0), -1)

    tp = (pred_f * target_f).sum(dim=1)
    fp = (pred_f * (1 - target_f)).sum(dim=1)
    fn = ((1 - pred_f) * target_f).sum(dim=1)
    tn = ((1 - pred_f) * (1 - target_f)).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice, iou, acc


def hausdorff_distance(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    dt_gt = distance_transform_edt(np.logical_not(gt))
    dt_pred = distance_transform_edt(np.logical_not(pred))

    return float(max(dt_gt[pred].max(), dt_pred[gt].max()))


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    dice_list = []
    iou_list = []
    acc_list = []
    hd_list = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        dice, iou, acc = dice_iou_acc_from_logits(
            logits,
            masks,
            threshold=threshold
        )

        dice_list.append(dice.detach().cpu().numpy())
        iou_list.append(iou.detach().cpu().numpy())
        acc_list.append(acc.detach().cpu().numpy())

        pred_np = (torch.sigmoid(logits).detach().cpu().numpy() > threshold)
        gt_np = (masks.detach().cpu().numpy() > 0.5)

        for i in range(pred_np.shape[0]):
            hd_list.append(
                hausdorff_distance(pred_np[i, 0], gt_np[i, 0])
            )

    dice_all = np.concatenate(dice_list, axis=0)
    iou_all = np.concatenate(iou_list, axis=0)
    acc_all = np.concatenate(acc_list, axis=0)
    hd_all = np.array(hd_list, dtype=np.float32)

    return {
        "mean_dice": float(dice_all.mean()),
        "mean_iou": float(iou_all.mean()),
        "mean_acc": float(acc_all.mean()),
        "mean_hd": float(hd_all.mean()),
    }


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        loss = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().item()))

    return float(np.mean(losses)) if losses else 0.0


class EarlyStop:
    def __init__(self, patience=10, min_delta=1e-4):
        self.best = -float("inf")
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def step(self, current):
        if current > self.best + self.min_delta:
            self.best = current
            self.counter = 0
            return False, True

        self.counter += 1
        return self.counter >= self.patience, False


def build_model(name):
    name = name.lower()

    if name == "unet":
        return UNet2D(in_channels=SEG_IN_CHANNELS, num_classes=1)

    if name == "vnet":
        return VNet2D(in_channels=SEG_IN_CHANNELS, num_classes=1)

    if name == "r50_unet":
        return R50UNet(in_channels=SEG_IN_CHANNELS, num_classes=1)

    if name == "deeplabv3":
        return DeepLabV3Binary(in_channels=SEG_IN_CHANNELS)

    raise ValueError(f"Unknown model: {name}")

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Real train set
    real_train = CellDataset(
        os.path.join(ROOT, "train", "images"),
        os.path.join(ROOT, "train", "masks"),
        image_size=IMG_SIZE
    )

    # Fake train set exported from GAN
    fake_train = CellDataset(
        os.path.join(EXPORT_ROOT, "images"),
        os.path.join(EXPORT_ROOT, "masks"),
        image_size=IMG_SIZE
    )

    train_dataset = ConcatDataset([real_train, fake_train])

    print(f"REAL TRAIN: {len(real_train)}")
    print(f"FAKE TRAIN: {len(fake_train)}")
    print(f"TOTAL TRAIN: {len(train_dataset)}")

    # Original validation set
    val_dataset = CellDataset(
        os.path.join(ROOT, "val", "images"),
        os.path.join(ROOT, "val", "masks"),
        image_size=IMG_SIZE
    )

    # Original test set
    test_dataset = CellDataset(
        os.path.join(ROOT, "test", "images"),
        os.path.join(ROOT, "test", "masks"),
        image_size=IMG_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=SEG_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    MODEL_NAME = "vnet"  #change model

    model = build_model(MODEL_NAME).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=SEG_LR,
        weight_decay=SEG_WEIGHT_DECAY
    )

    save_dir = os.path.join(SEG_SAVE_PATH, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"best_{MODEL_NAME}.pth")

    early = EarlyStop(
        patience=SEG_EARLY_STOP_PATIENCE,
        min_delta=SEG_EARLY_STOP_MIN_DELTA
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_acc": [],
        "val_hd": [],
        "best_val_dice_sofar": [],
        "improved": [],
    }

    best_epoch = None
    stop_epoch = None

    print(f"\nTraining {MODEL_NAME.upper()} on REAL + FAKE train set")

    for epoch in range(SEG_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=SEG_THRESHOLD
        )

        print(f"\n[Epoch {epoch + 1}/{SEG_EPOCHS}] train_loss={train_loss:.4f}")
        print(
            f"VAL Dice={val_metrics['mean_dice']:.4f} | "
            f"IoU={val_metrics['mean_iou']:.4f} | "
            f"Acc={val_metrics['mean_acc']:.4f} | "
            f"HD={val_metrics['mean_hd']:.2f}px"
        )

        should_stop, improved = early.step(val_metrics["mean_dice"])

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_metrics["mean_dice"])
        history["val_iou"].append(val_metrics["mean_iou"])
        history["val_acc"].append(val_metrics["mean_acc"])
        history["val_hd"].append(val_metrics["mean_hd"])
        history["best_val_dice_sofar"].append(float(early.best))
        history["improved"].append(bool(improved))

        if improved:
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": float(early.best),
                    "history": history,
                    "in_channels": SEG_IN_CHANNELS,
                },
                best_path
            )

            print(f"Saved BEST {MODEL_NAME.upper()} (val Dice={early.best:.4f})")
        else:
            print(
                f"No improvement: "
                f"{early.counter}/{early.patience} "
                f"(best Dice={early.best:.4f})"
            )

        if should_stop:
            stop_epoch = epoch + 1
            print(
                f"\nEarly stopping at epoch {epoch + 1}. "
                f"Best val Dice={early.best:.4f}"
            )
            break

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"\nLoaded best {MODEL_NAME.upper()} checkpoint:", best_path, "| best_val_dice=", checkpoint["best_val_dice"])
    else:
        print("\nNo best checkpoint found. Testing current model.")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=SEG_THRESHOLD
    )

    print(f"\n===== TEST RESULT: {MODEL_NAME.upper()} REAL + FAKE TRAIN =====")
    print(f"mean Dice = {test_metrics['mean_dice']:.4f}")
    print(f"mean IoU  = {test_metrics['mean_iou']:.4f}")
    print(f"mean Acc  = {test_metrics['mean_acc']:.4f}")
    print(f"mean HD   = {test_metrics['mean_hd']:.2f}px")

    print("\n===== TRAIN SUMMARY =====")
    print("Best epoch:", best_epoch)
    print("Early stop epoch:", stop_epoch)
    print("Best val dice:", max(history["val_dice"]) if history["val_dice"] else None)



if __name__ == "__main__":
    main()