import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from configs.defaults import USE_NOISE, MASK_THRESHOLD
from src.utils import build_inputs_for_G


@torch.no_grad()
def export_fake_image_and_fake_mask(
    image_generator,
    mask_generator,
    loader,
    export_root,
    device,
    limit=None
):
    image_dir = os.path.join(export_root, "images")
    mask_dir = os.path.join(export_root, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_generator.eval()
    mask_generator.eval()

    saved = 0

    for batch in tqdm(loader, desc="Exporting fake image + fake mask"):
        real_image = batch["image"].to(device).float()
        real_mask = batch["mask"].to(device).float()
        ids = batch["id"]

        # Branch 1: real image -> fake mask
        mask_logits = mask_generator(real_image)
        mask_probs = torch.sigmoid(mask_logits)
        fake_mask = (mask_probs >= MASK_THRESHOLD).float()

        # Branch 2: real mask + noise -> fake image
        fake_image = image_generator(
            build_inputs_for_G(real_mask, use_noise=USE_NOISE)
        )

        fake_image = ((fake_image + 1) / 2).clamp(0, 1)

        for i in range(fake_image.size(0)):
            sample_id = str(ids[i])

            # save fake image
            img = fake_image[i].detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255.0).round().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(
                os.path.join(image_dir, f"image_{sample_id}_img.png"),
                img
            )

            # save fake mask
            mask = fake_mask[i].detach().cpu().numpy().squeeze(0)
            mask = (mask * 255.0).round().astype(np.uint8)

            cv2.imwrite(
                os.path.join(mask_dir, f"image_{sample_id}_masks.png"),
                mask
            )

            saved += 1

            if limit is not None and saved >= limit:
                print(f"Exported {saved} samples to: {export_root}")
                return

    print(f"Exported {saved} samples to: {export_root}")