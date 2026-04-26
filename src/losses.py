import torch
import torch.nn.functional as F
from src.utils import rgb_to_gray, gaussian_blur


# =========================
# Basic Loss
# =========================
l1_loss = torch.nn.L1Loss()


# =========================
# Sobel Edge
# =========================
def sobel_edges(x):
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1],
         [0,  0,  0],
         [1,  2,  1]],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)

    return torch.sqrt(gx * gx + gy * gy + 1e-12)


# =========================
# Edge Loss (Foreground)
# =========================
def edge_loss_fg(fake_rgb, real_rgb, mask):
    f = rgb_to_gray(fake_rgb)
    r = rgb_to_gray(real_rgb)

    f01 = ((f + 1) / 2).clamp(0, 1)
    r01 = ((r + 1) / 2).clamp(0, 1)

    ef = sobel_edges(f01)
    er = sobel_edges(r01)

    m = (mask > 0.5).float()

    return F.l1_loss(ef * m, er * m)


# =========================
# High-pass Loss (Foreground)
# =========================
def hp_loss_fg(fake_rgb, real_rgb, mask):
    f = rgb_to_gray(fake_rgb)
    r = rgb_to_gray(real_rgb)

    hp_f = f - gaussian_blur(f, 7, 3)
    hp_r = r - gaussian_blur(r, 7, 3)

    m = (mask > 0.5).float()

    return F.l1_loss(hp_f * m, hp_r * m)


# =========================
# Masked Mean + Std
# =========================
def masked_mean_std(x, m, eps=1e-6):
    w = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

    mean = (x * m).sum(dim=(2, 3), keepdim=True) / w
    var = ((x - mean) ** 2 * m).sum(dim=(2, 3), keepdim=True) / w
    std = torch.sqrt(var + eps)

    return mean, std


# =========================
# Intensity Matching Loss
# =========================
def intensity_match_loss(fake_rgb, real_rgb, mask):
    f = rgb_to_gray(fake_rgb)
    r = rgb_to_gray(real_rgb)

    f01 = ((f + 1) / 2).clamp(0, 1)
    r01 = ((r + 1) / 2).clamp(0, 1)

    fg = (mask > 0.5).float()
    bg = 1 - fg

    mf_mean, mf_std = masked_mean_std(f01, fg)
    mr_mean, mr_std = masked_mean_std(r01, fg)

    mbf_mean, mbf_std = masked_mean_std(f01, bg)
    mbr_mean, mbr_std = masked_mean_std(r01, bg)

    return (
        torch.abs(mf_mean - mr_mean).mean() +
        torch.abs(mf_std - mr_std).mean() +
        torch.abs(mbf_mean - mbr_mean).mean() +
        torch.abs(mbf_std - mbr_std).mean()
    )


# =========================
# Low-pass Loss
# =========================
def low_pass_loss(fake_rgb, real_rgb):
    return l1_loss(
        gaussian_blur(fake_rgb, 21, 11),
        gaussian_blur(real_rgb, 21, 11)
    )


# =========================
# GAN Loss (Hinge)
# =========================
def d_hinge_loss(pr, pf):
    return F.relu(1.0 - pr).mean() + F.relu(1.0 + pf).mean()


def g_hinge_loss(pf):
    return -pf.mean()


# =========================
# Combined Loss Parts
# =========================
def compute_loss_parts(fake, real, mask,
                       lambda_l1_fg,
                       lambda_l1_bg,
                       lambda_edge_fg,
                       lambda_hp_fg,
                       lambda_intm,
                       lambda_tv_fg,
                       lambda_low,
                       total_variation_loss_fg):

    mask3 = mask.repeat(1, 3, 1, 1)
    bg3 = (1 - mask).repeat(1, 3, 1, 1)

    # L1 losses
    l1_fg = l1_loss(fake * mask3, real * mask3)
    l1_bg = l1_loss(fake * bg3, real * bg3)

    # Structural losses
    ed_fg = edge_loss_fg(fake, real, mask)
    hp_fg = hp_loss_fg(fake, real, mask)
    intm = intensity_match_loss(fake, real, mask)

    # Optional losses
    tv_fg = total_variation_loss_fg(fake, mask) if lambda_tv_fg > 0 else torch.tensor(0.0, device=fake.device)
    low = low_pass_loss(fake, real) if lambda_low > 0 else torch.tensor(0.0, device=fake.device)

    # Combined generator-only loss
    genonly = (
        lambda_l1_fg * l1_fg +
        lambda_l1_bg * l1_bg +
        lambda_edge_fg * ed_fg +
        lambda_hp_fg * hp_fg +
        lambda_intm * intm +
        lambda_tv_fg * tv_fg +
        lambda_low * low
    )

    return {
        "l1_fg": l1_fg,
        "l1_bg": l1_bg,
        "edge_fg": ed_fg,
        "hp_fg": hp_fg,
        "intm": intm,
        "tv_fg": tv_fg,
        "low": low,
        "genonly": genonly
    }

# =========================
# Mask Metrics
# =========================
def dice_coefficient_from_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * inter + smooth) / (denom + smooth)
    return dice.mean()


def dice_loss_from_logits(logits, targets, smooth=1e-6):
    return 1.0 - dice_coefficient_from_logits(logits, targets, smooth)


def iou_from_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter

    iou = (inter + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    return (preds == targets).float().mean()


# =========================
# Mask Edge Loss
# =========================
def edge_loss_mask(logits, targets):
    probs = torch.sigmoid(logits)

    pred_edges = sobel_edges(probs)
    target_edges = sobel_edges(targets)

    return F.l1_loss(pred_edges, target_edges)


# =========================
# Total Mask Loss
# =========================
def total_mask_loss(
    logits,
    targets,
    lambda_bce=1.0,
    lambda_dice=1.0,
    lambda_edge=0.30
):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    edge = edge_loss_mask(logits, targets)

    total = (
        lambda_bce * bce +
        lambda_dice * dice +
        lambda_edge * edge
    )

    return total, {
        "bce": bce,
        "dice_loss": dice,
        "edge": edge
    }