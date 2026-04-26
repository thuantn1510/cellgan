import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Normalization (GroupNorm)
# =========================
def Norm(ch):
    return nn.GroupNorm(num_groups=8, num_channels=ch)


# =========================
# Attention Gate
# =========================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            Norm(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            Norm(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # align size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# =========================
# Residual Block
# =========================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            Norm(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            Norm(ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


# =========================
# Encoder Block
# =========================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            ResBlock(out_ch)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


# =========================
# Decoder Block
# =========================
class DecoderBlock(nn.Module):
    def __init__(self, skip_ch, up_ch, out_ch):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(up_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True)
        )

        self.att = AttentionGate(out_ch, skip_ch, out_ch // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            ResBlock(out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        skip = self.att(x, skip)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =========================
# Generator (ResUNet + Attention)
# =========================
class ResUNetWithAttention(nn.Module):
    def __init__(self, in_ch=2, out_ch=3):
        super().__init__()

        self.enc1 = EncoderBlock(in_ch, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 1024)

        self.bottleneck = ResBlock(1024)

        self.dec4 = DecoderBlock(512, 1024, 512)
        self.dec3 = DecoderBlock(256, 512, 256)
        self.dec2 = DecoderBlock(128, 256, 128)
        self.dec1 = DecoderBlock(64, 128, 64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_ch, 1),
            nn.Tanh()
        )

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        s5, p5 = self.enc5(p4)

        b = self.bottleneck(p5)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


# =========================
# PatchGAN Discriminator
# =========================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()

        def blk(ci, co, stride=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, 4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(co, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            blk(64, 128),
            blk(128, 256),
            blk(256, 512),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Mask Generator
# ResUNet + Attention Gate
# Input : image [B,3,H,W]
# Output: mask logits [B,1,H,W]
# =========================
class ResUNetWithAttentionMask(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.enc1 = EncoderBlock(in_ch, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 1024)

        self.bottleneck = ResBlock(1024)

        self.dec4 = DecoderBlock(512, 1024, 512)
        self.dec3 = DecoderBlock(256, 512, 256)
        self.dec2 = DecoderBlock(128, 256, 128)
        self.dec1 = DecoderBlock(64, 128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        s5, p5 = self.enc5(p4)

        b = self.bottleneck(p5)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)