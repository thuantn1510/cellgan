# GAN-based Data Augmentation for Cell Image Segmentation

## Overview

Cell image segmentation is an important task in many biomedical image analysis applications. However, the limited availability of annotated datasets can reduce the performance of deep learning-based segmentation models. This work proposes a GAN-based data augmentation framework to improve cell image segmentation performance. The proposed approach generates cell images using an improved conditional generative adversarial network in which a residual U-Net architecture with Attention mechanisms is employed to produce cell images conditioned on cell masks. Besides, a noise injection technique is also applied to improve the augmentation quality

## Dataset

This project is designed to work with paired cell image datasets structured in a specific format. You can place your dataset (e.g., BBBC038v1) inside the `data/` directory.

Please organize your dataset within the `data/` directory as follows:

```text
data/
└── your_dataset_name/   # e.g., BBBC038v1
    ├── train/
    │   ├── images/      # Training images (e.g., .png)
    │   └── masks/       # Training masks (e.g., .png)
    ├── val/
    │   ├── images/      # Validation images
    │   └── masks/       # Validation masks
    └── test/
        ├── images/      # Test images
        └── masks/       # Test masks
```
Important: Update the dataset path in configs/defaults.py to point to your dataset directory: 
`ROOT = os.path.join("data", "BBBC038v1")`

## Project Structure

```text
├── data/
│   └── BBBC038v1/              # Dataset directory (train/val/test)
│
├── src/                        # Core source code modules
│   ├── dataset.py              # Dataset loader (CellDataset)
│   ├── model.py                # Generator (ResUNet + Attention Gate) and Discriminator
│   ├── losses.py               # GAN losses and auxiliary losses
│   ├── utils.py                # Helper functions (noise injection, etc.)
│   ├── engine.py               # Training pipeline for image generator (GAN)
│   ├── mask_engine.py          # Training pipeline for mask generator
│   ├── export.py               # Export synthetic image–mask pairs
│   ├── train.py                # Train GAN and generate synthetic dataset
│   ├── train_mask.py           # Train mask generator
│   └── train_segmentation.py.py # Train segmentation models (U-Net, V-Net, R50-U-Net, DeepLabV3)
│
├── configs/
│   └── defaults.py             # Default hyperparameters and dataset paths
│
├── .gitignore                  # Files/folders ignored by Git
├── README.md                   # Project overview and instructions

## Contact

Tran Nhu Thuan  
thuantn1510@gmail.com