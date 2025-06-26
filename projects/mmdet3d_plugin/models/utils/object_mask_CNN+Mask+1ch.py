import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

def get_bev_map(space3d_net, x):
    _, bev_feat = space3d_net.forward(x, bev_only=True)
    return bev_feat

class ObjectMaskNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64): #3-layer CNN
        super(ObjectMaskNet, self).__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 마스크 값을 0~1로 정규화
        )

    def forward(self, bev_feat):
        return self.mask_conv(bev_feat)


class ObjectMaskModule(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, detach_mask=True, epoch_threshold=1): 
        super(ObjectMaskModule, self).__init__()
        self.object_mask_net = ObjectMaskNet(in_channels, hidden_dim)    

    def forward(self, bev_feat):
        object_mask = self.object_mask_net(bev_feat)  # (B, 1, H, W)
        # print("\n     #1 [object_mask.py: ObjectMaskModule.forward] Before detach, object_mask.shape:", object_mask.shape)
        # print("     #2 [object_mask.py: ObjectMaskModule.forward] Detaching object_mask")
        return object_mask