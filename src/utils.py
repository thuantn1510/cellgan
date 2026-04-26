import torch
import torch.nn.functional as F


# =========================
# Basic image ops
# =========================
def rgb_to_gray(x):
    """
    Convert RGB image to grayscale
    x: (B,3,H,W)
    """
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gaussian_blur(x, kernel_size=5, sigma=1.0):
    """
    Simple Gaussian blur using conv
    """
    device = x.device
    channels = x.shape[1]

    # Create kernel
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=channels)


# =========================
# Noise Injection (QUAN TRỌNG)
# =========================
def generate_noise(mask, strength=1.0):
    """
    Multi-scale noise injection theo đúng ý tưởng của bạn
    """
    B, _, H, W = mask.shape
    device = mask.device

    z_low = torch.randn(B, 1, H, W, device=device)
    z_mid = torch.randn(B, 1, H, W, device=device)
    z_hi  = torch.randn(B, 1, H, W, device=device)

    noise = (
        0.10 * mask * z_low +
        (1 - mask) * (0.25 * z_low + 0.08 * z_mid + 0.03 * z_hi)
    )

    return strength * noise


# =========================
# Input Builder
# =========================
def build_inputs_for_G(mask, use_noise=True):
    """
    Build input for Generator
    Output shape: (B,2,H,W)
    """
    if use_noise:
        noise = generate_noise(mask)
    else:
        noise = torch.zeros_like(mask)

    return torch.cat([mask, noise], dim=1)


def build_input_for_D(image, mask):
    """
    Input for Discriminator
    Output shape: (B,4,H,W)
    """
    return torch.cat([image, mask], dim=1)


# =========================
# Loss helpers
# =========================
def total_variation_loss_fg(img, mask):
    """
    TV loss chỉ tính trên foreground
    """
    diff_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    diff_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    mask_x = mask[:, :, :, :-1]
    mask_y = mask[:, :, :-1, :]

    return (diff_x * mask_x).mean() + (diff_y * mask_y).mean()


def assert_finite(tensor, name="tensor"):
    """
    Debug NaN/Inf
    """
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf!")