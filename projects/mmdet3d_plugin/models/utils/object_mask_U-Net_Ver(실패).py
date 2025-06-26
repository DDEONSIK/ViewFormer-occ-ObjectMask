# 현재 object_mask.py 전체 코드:
# 위치: ViewFormer-Occ/projects/mmdet3d_plugin/models/utils/object_mask.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def get_bev_map(space3d_net, x):
    """
    BEV Feature(B_t)를 추출하는 함수
    """
    _, bev_feat = space3d_net.forward(x, bev_only=True)
    return bev_feat

class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=False)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=False)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=False)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, ceil_mode=False)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("▶ INPUT x shape:", x.shape)
        enc1 = self.enc1(x)
        # print("▶ enc1:", enc1.shape)
        enc2 = self.enc2(self.pool1(enc1))
        # print("▶ enc2:", enc2.shape)
        enc3 = self.enc3(self.pool2(enc2))
        # print("▶ enc3:", enc3.shape)
        enc4 = self.enc4(self.pool3(enc3))
        # print("▶ enc4:", enc4.shape)
        bottleneck = self.bottleneck(self.pool4(enc4))
        # print("▶ bottleneck:", bottleneck.shape)

        # dec4 = self.upconv4(bottleneck)
        dec4 = self.upconv4(bottleneck).contiguous()
        # print("▶ dec4 (before cat):", dec4.shape)
        enc4_resized = F.interpolate(enc4, size=dec4.shape[2:], mode='bilinear', align_corners=False).contiguous()
        dec4 = self.dec4(torch.cat((dec4, enc4_resized), dim=1))

        # dec3 = self.upconv3(dec4)
        dec3 = self.upconv3(dec4).contiguous()
        # print("▶ dec3 (before cat):", dec3.shape)
        enc3_resized = F.interpolate(enc3, size=dec3.shape[2:], mode='bilinear', align_corners=False).contiguous()
        dec3 = self.dec3(torch.cat((dec3, enc3_resized), dim=1))

        # dec2 = self.upconv2(dec3)
        dec2 = self.upconv2(dec3).contiguous()
        # print("▶ dec2 (before cat):", dec2.shape)
        enc2_resized = F.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=False).contiguous()
        dec2 = self.dec2(torch.cat((dec2, enc2_resized), dim=1))

        # dec1 = self.upconv1(dec2)
        dec1 = self.upconv1(dec2).contiguous()
        # print("▶ dec1 (before cat):", dec1.shape)
        enc1_resized = F.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False).contiguous()
        dec1 = self.dec1(torch.cat((dec1, enc1_resized), dim=1))

        out = self.final_conv(dec1)
        out = F.interpolate(out, size=(200, 200), mode='bilinear', align_corners=False).contiguous()
        # print("▶ final output shape:", out.shape)
        return self.sigmoid(out)


class ObjectMaskNet(nn.Module):
    def __init__(self, in_channels):
        super(ObjectMaskNet, self).__init__()
        self.unet = UNet(in_channels)

    def forward(self, bev_feat):
        return self.unet(bev_feat)

def apply_object_mask(bev_feat, object_mask, threshold=0.5):
    binary_mask = (object_mask > threshold).float()
    return bev_feat * binary_mask

class ObjectMaskModule(nn.Module):
    def __init__(self, in_channels):
        super(ObjectMaskModule, self).__init__()
        self.object_mask_net = ObjectMaskNet(in_channels)

    def set_epoch(self, epoch):
        pass

    def forward(self, bev_feat):
        object_mask = self.object_mask_net(bev_feat)  # (B, 1, H, W)
        # print("\n     #1 [object_mask.py: ObjectMaskModule.forward] Before detach, object_mask.shape:", object_mask.shape)
        # print("     #2 [object_mask.py: ObjectMaskModule.forward] Detaching object_mask")
        return object_mask
