# GAN-based Data Augmentation for Cell Image Segmentation

## Overview

Cell image segmentation is an important task in many biomedical image analysis applications. However, the limited availability of annotated datasets can reduce the performance of deep learning-based segmentation models. This work proposes a GAN-based data augmentation framework to improve cell image segmentation performance. The proposed approach generates cell images using an improved conditional generative adversarial network in which a residual U-Net architecture with Attention mechanisms is employed to produce cell images conditioned on cell masks. Besides, a noise injection technique is also applied to improve the augmentation quality

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/thuannt1510/cellgan.git
cd cellgan