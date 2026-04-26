# GAN-based Data Augmentation for Cell Image Segmentation

## Overview

This repository contains the implementation of a GAN-based data augmentation framework for cell image segmentation. The proposed method generates synthetic cell image–mask pairs to enlarge the training set and improve segmentation performance.

The framework consists of two main stages. First, a mask generator based on ResUNet with Attention Gate is trained to generate synthetic masks from real training images. Second, a conditional GAN is used to generate synthetic cell images from masks and structured noise. The generated image–mask pairs are then combined with the original training set to train segmentation models.

The segmentation performance is evaluated using U-Net, V-Net, R50-U-Net, and DeepLabV3 with Dice, IoU, Accuracy, and Hausdorff Distance.

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/thuannt1510/cellgan.git
cd cellgan