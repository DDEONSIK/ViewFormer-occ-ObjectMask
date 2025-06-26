from .mix_transformer import MixVisionTransformer
from .segformer_head import SegFormerHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial 

class SegFormerMaskNet(nn.Module):
    def __init__(self, bev_channels=126, embed_dims=[64, 128, 320, 512], num_classes=1):
        super(SegFormerMaskNet, self).__init__()

            # # BEV 피처에서 multi-scale feature 생성
            # self.scale1 = nn.Sequential(
            #     nn.Conv2d(bev_channels, embed_dims[0], kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True)
            # ) # 1/1 scale 유지 (예: 200x200)

            # self.scale2 = nn.Sequential(
            #     nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1),
            #     nn.ReLU(inplace=True)
            # ) # 1/2 scale (예: 100x100)

            # self.scale3 = nn.Sequential(
            #     nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1),
            #     nn.ReLU(inplace=True)
            # ) # 1/4 scale (예: 50x50)

            # self.scale4 = nn.Sequential(
            #     nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1),
            #     nn.ReLU(inplace=True)
            # ) # 1/8 scale (예: 25x25)
        # print(" ---- [SegFormerMaskNet] Initialized with bev_channels:", bev_channels)
        # SegFormer: Transformer MiT backbone (SegFormer 원본 구조)
        self.transformer = MixVisionTransformer(
            img_size=200,           # BEV feature의 해상도에 맞춰야 함 
            patch_size=4,
            in_chans=bev_channels,
            embed_dims=embed_dims,             # [64, 128, 320, 512]
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                # 아래 Depth는 4개의 Transformer 인코더 스테이지별 Transformer block 레이어 수를 의미
                # depths=[2, 2, 2, 2],      # SegFormer-B0,1
                # depths=[3, 4, 6, 3],      # SegFormer-B2
                # depths=[3, 4, 18, 3],      # SegFormer-B3
            # depths=[3, 8, 27, 3],      # SegFormer-B4
            depths=[3, 6, 40, 3],       # SegFormer-B5
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )

        # MLP Decode Head (SegFormer Head)
        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128, # decode head 채널
            dropout_ratio=0.1,
            num_classes=num_classes,
            align_corners=False,
            decoder_params=dict(embed_dim=128)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feat):
                    # # BEV에서 multi-scale 피처 생성
                    # s1 = self.scale1(bev_feat)      # [B,64,200,200]
                    # s2 = self.scale2(s1)            # [B,128,100,100]
                    # s3 = self.scale3(s2)            # [B,320,50,50]
                    # s4 = self.scale4(s3)            # [B,512,25,25]

                    # # 디버깅: 각 scale feature sum
                    # print("[DEBUG] s1.shape:", s1.shape)
                    # print("[DEBUG] s2.sum():", s2.sum().item())
                    # print("[DEBUG] s3.sum():", s3.sum().item())
                    # print("[DEBUG] s4.sum():", s4.sum().item())

                    # # transformer_feats는 s1 하나만 받는다.
                    # # transformer_feats = self.transformer.forward_features(s1)
        transformer_feats = self.transformer.forward_features(bev_feat)  # [x1, x2, x3, x4]

            # decode head 입력에 각 scale feature를 연결하는 경우는 다음과 같이 교체 가능
            # 만약 decoder가 직접 연결하는게 아니라면 아래와 같이 명시적으로 전달
            # transformer_feats = [s1, s2, s3, s4]  # 주의: 구조 변경 필요 시 decode_head도 조정해야 함

        # for i, feat in enumerate(transformer_feats):
        #     print(f"[DEBUG] transformer_feats[{i}].shape:", feat.shape)
            # torch.Size([1, 64, 50, 50])
            # torch.Size([1, 128, 25, 25])
            # torch.Size([1, 320, 13, 13])
            # torch.Size([1, 512, 7, 7])

        # [중요] conv_seg 파라미터 상태 출력
        # print("[CHECKPOINT] BEFORE seg_logits 계산 ================================")
        # print("  conv_seg.weight.requires_grad:", self.decode_head.conv_seg.weight.requires_grad) # True
        # print("  conv_seg.bias.requires_grad:", self.decode_head.conv_seg.bias.requires_grad) # True
        # print("  ⚠️   conv_seg.weight.grad:", self.decode_head.conv_seg.weight.grad) # None
        # print("  ⚠️   conv_seg.bias.grad:", self.decode_head.conv_seg.bias.grad) # None


        # 마스크 예측
        seg_logits = self.decode_head(transformer_feats, return_logits=True)
        # print("[DEBUG] seg_logits.shape:", seg_logits.shape) # torch.Size([1, 1, 50, 50])
        _ = torch.sum(seg_logits) * 1e-7
            # mask_pred = seg_logits

            # # 출력 크기 정렬
            # mask_pred = F.interpolate(mask_pred, size=bev_feat.shape[2:], mode='bilinear', align_corners=False)
            # mask_pred = self.sigmoid(mask_pred)

        # print("[CHECKPOINT] AFTER seg_logits 계산 --------------------------------")
        # print("  seg_logits.requires_grad:", seg_logits.requires_grad) # True
        # print("  seg_logits.grad_fn:", seg_logits.grad_fn) # <ConvolutionBackward0 object at 0x7ff618015c10>
        # print("===============================================================")

        return seg_logits


class ObjectMaskModule(nn.Module):
    def __init__(self, in_channels=126, detach_mask=True, epoch_threshold=1):
        super(ObjectMaskModule, self).__init__()
        self.object_mask_net = SegFormerMaskNet(bev_channels=in_channels)

    def forward(self, bev_feat):
        # print("[FINAL DEBUG] SegFormer input requires_grad:", bev_feat.requires_grad)

        # bev_feat.detach() 라인을 제거하여 상위 네트워크로부터 그래디언트가 흐르도록 함
        # with torch.set_grad_enabled(self.training): # 이 블록은 self.training 상태에 따라 내부 모듈의 그래디언트 계산 여부를 제어
        # ObjectMaskNet (SegFormerMaskNet)이 nn.Module을 상속받고,
        # ObjectMaskModule의 training 상태가 ObjectMaskNet에 올바르게 전달된다면
        # (일반적으로 부모 모듈의 train()/eval() 호출 시 자식 모듈도 따라서 변경됨),
        # 이 with 블록은 필수적이지 않을 수 있으나, 명시적으로 두는 것도 안전합니다.
        # 여기서는 with 블록을 유지하되, bev_feat.detach()만 제거하는 것으로 가정합니다.
        
        # 만약 ObjectMaskModule이 eval 모드일 때도 내부 연산이 동일해야 한다면,
        # with torch.set_grad_enabled(True) 또는 상황에 맞게 조정 필요.
        # 일반적으로 학습 시에는 self.training이 True이므로 그래디언트가 잘 계산됩니다.

        # bev_feat가 detach 되지 않았으므로, 여기서부터 계산되는 seg_logits는
        # bev_feat 및 그 이전 레이어들과의 연결성을 유지합니다.

        # -> Detach를 제거 실험: 
        transformer_feats = self.object_mask_net.transformer.forward_features(bev_feat)
        seg_logits = self.object_mask_net.decode_head(transformer_feats, return_logits=True)


        # -> 논문:
        # bev_feat = bev_feat.detach()
        # with torch.set_grad_enabled(self.training):
        #     transformer_feats = self.object_mask_net.transformer.forward_features(bev_feat)
        #     seg_logits = self.object_mask_net.decode_head(transformer_feats, return_logits=True)


        # print("[DEBUG] object_mask.forward() → seg_logits.requires_grad:", seg_logits.requires_grad)
        # print("[DEBUG] object_mask.forward() → seg_logits.grad_fn:", seg_logits.grad_fn)

        return seg_logits