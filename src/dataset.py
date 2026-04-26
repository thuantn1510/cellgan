import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CellDataset(Dataset):
    IMG_RE = re.compile(r"^image_(\d+)_img\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)
    MASK_RE = re.compile(r"^image_(\d+)_masks\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)

    def __init__(self, images_dir, masks_dir, image_size=(512, 768)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5]*3, std=[0.5]*3)

        img_map = {}
        mask_map = {}

        for f in os.listdir(images_dir):
            m = self.IMG_RE.match(f)
            if m:
                img_map[m.group(1)] = f

        for f in os.listdir(masks_dir):
            m = self.MASK_RE.match(f)
            if m:
                mask_map[m.group(1)] = f

        self.ids = sorted(
            list(set(img_map.keys()) & set(mask_map.keys())),
            key=lambda x: int(x)
        )

        self.img_map = img_map
        self.mask_map = mask_map

        if len(self.ids) == 0:
            raise RuntimeError("No matching image-mask pairs found!")

        print(f"Loaded {len(self.ids)} samples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        # ===== Load Image =====
        img_path = os.path.join(self.images_dir, self.img_map[sample_id])
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size[::-1])
        img = img.astype(np.float32) / 255.0

        # ===== Load Mask =====
        mask_path = os.path.join(self.masks_dir, self.mask_map[sample_id])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(mask, self.image_size[::-1])
        mask = (mask > 0).astype(np.float32)

        # ===== To Tensor =====
        img = self.normalize(self.to_tensor(img))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": img,
            "mask": mask,
            "id": sample_id
        }

class ImageOnlyDataset(Dataset):
    IMG_RE = re.compile(r"^image_(\d+)_img\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)

    def __init__(self, images_dir, image_size=(512, 768)):
        self.images_dir = images_dir
        self.image_size = image_size

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.files = []
        self.ids = []

        for f in sorted(os.listdir(images_dir)):
            m = self.IMG_RE.match(f)
            if m:
                self.files.append(f)
                self.ids.append(m.group(1))

        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in: {images_dir}")

        print(f"Found {len(self.files)} images in {images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        sample_id = self.ids[idx]

        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(img_path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(
            img,
            self.image_size[::-1],
            interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0

        img = self.normalize(self.to_tensor(img))

        return {
            "image": img,
            "id": sample_id,
            "filename": fname
        }