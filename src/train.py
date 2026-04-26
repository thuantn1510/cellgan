import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import ResUNetWithAttentionMask
from src.export import export_fake_image_and_fake_mask


from configs.defaults import *

from src.dataset import CellDataset
from src.model import ResUNetWithAttention, PatchDiscriminator
from src.utils import (
    build_inputs_for_G,
    build_input_for_D,
    assert_finite,
    total_variation_loss_fg
)
from src.losses import (
    compute_loss_parts,
    d_hinge_loss,
    g_hinge_loss
)

# =========================
# Device
# =========================
DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# =========================
# Dataset Loader
# =========================
try:
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

except Exception as e:
    print(f"ERROR: Could not load dataset. Check ROOT in configs/defaults.py\n{e}")
    exit()


# =========================
# Models
# =========================
G = ResUNetWithAttention(
    in_ch=2 if USE_NOISE else 1,
    out_ch=3
).to(DEVICE)

D = PatchDiscriminator(in_ch=4).to(DEVICE)


# =========================
# Optimizers
# =========================
optG = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
optD = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)


# =========================
# Training Loop
# =========================
def train():

    for epoch in range(EPOCHS):

        G.train()
        D.train()

        # Warmup GAN
        cur_lambda_adv = 0.0 if epoch < WARMUP_EPOCHS else LAMBDA_ADV

        d_losses = []
        g_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            real = batch["image"].to(DEVICE).float()
            mask = batch["mask"].to(DEVICE).float()

            # ======================
            # Generator forward
            # ======================
            g_input = build_inputs_for_G(mask, use_noise=USE_NOISE)
            fake = G(g_input)

            if fake.shape != real.shape:
                fake = torch.nn.functional.interpolate(
                    fake,
                    size=real.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

            assert_finite(fake, "fake")

            # ======================
            # Train Discriminator
            # ======================
            if cur_lambda_adv > 0:
                real_input = build_input_for_D(real, mask)
                fake_input = build_input_for_D(fake.detach(), mask)

                pr = D(real_input)
                pf = D(fake_input)

                d_loss = d_hinge_loss(pr, pf)

                optD.zero_grad()
                d_loss.backward()
                optD.step()

                d_losses.append(d_loss.item())

            # ======================
            # Train Generator
            # ======================
            fake_input = build_input_for_D(fake, mask)

            adv_loss = (
                g_hinge_loss(D(fake_input))
                if cur_lambda_adv > 0
                else torch.tensor(0.0, device=DEVICE)
            )

            parts = compute_loss_parts(
                fake, real, mask,
                LAMBDA_L1_FG,
                LAMBDA_L1_BG,
                LAMBDA_EDGE_FG,
                LAMBDA_HP_FG,
                LAMBDA_INTM,
                LAMBDA_TV_FG,
                LAMBDA_LOW,
                total_variation_loss_fg
            )

            gen_loss = parts["genonly"]
            total_loss = gen_loss + cur_lambda_adv * adv_loss

            assert_finite(total_loss, "total_loss")

            optG.zero_grad()
            total_loss.backward()
            optG.step()

            g_losses.append(total_loss.item())

        # ======================
        # Logging
        # ======================
        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses) if d_losses else 0.0

        print(f"\nEpoch {epoch+1}")
        print(f"  Generator Loss: {avg_g:.4f}")
        if d_losses:
            print(f"  Discriminator Loss: {avg_d:.4f}")

    print("Training Done!")


if __name__ == "__main__":

    # =========================
    # Train GAN
    # =========================
    train()

    # =========================
    # Load best Image Generator
    # =========================
    best_img_ckpt = os.path.join(SAVE_PATH, "best_SEGSYN.pth")

    if os.path.exists(best_img_ckpt):
        print("Loading best image generator:", best_img_ckpt)
        ckpt = torch.load(best_img_ckpt, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt["generator_state_dict"])
    else:
        print("best_SEGSYN.pth not found. Using current generator.")

    # =========================
    # Load best Mask Generator
    # =========================
    G_mask = ResUNetWithAttentionMask(in_ch=3, out_ch=1).to(DEVICE)

    best_mask_ckpt = os.path.join(MASK_SAVE_PATH, "best_DICE.pth")

    if os.path.exists(best_mask_ckpt):
        print("Loading best mask generator:", best_mask_ckpt)
        ckpt = torch.load(best_mask_ckpt, map_location=DEVICE, weights_only=False)
        G_mask.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError(
            f"Mask checkpoint not found: {best_mask_ckpt}\n"
            "Run: python -m src.train_mask first"
        )

    # =========================
    # Export (train set only)
    # =========================
    export_fake_image_and_fake_mask(
        image_generator=G,
        mask_generator=G_mask,
        loader=train_loader,
        export_root=EXPORT_ROOT,
        device=DEVICE,
        limit=EXPORT_LIMIT
    )

    