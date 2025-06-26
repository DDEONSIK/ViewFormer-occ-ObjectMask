# 파일: projects/mmdet3d_plugin/models/utils/object_mask_loss.py
import torch
import torch.nn.functional as F

def loss_object_mask(pred_mask, gt_mask, loss_weight=1.0, smooth=1.0):
    """
    예측 객체 마스크와 GT 객체 마스크 간의 Binary Cross-Entropy + Dice Loss 계산.

    Args:
        pred_mask (Tensor): 예측 객체 마스크, shape: (B, C, H, W)
        gt_mask (Tensor): GT 객체 마스크, shape: (B, C, H, W)
        loss_weight (float): 손실 가중치
        smooth (float): Dice loss smoothing factor
    
    Returns:
        Tensor: 최종 계산된 객체 마스크 손실 값
    """
    
    # 논문:
    # # Binary Cross-Entropy Loss (채널별 평균)
    # if gt_mask.dim() == 3:
    #     gt_mask = gt_mask.unsqueeze(1) # (B, 1, H, W) 형식으로 변환
    # bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='mean')

    # # Dice Loss 계산 (채널별)
    # pred_flat = pred_mask.flatten(2)  # (B, C, H*W)
    # gt_flat = gt_mask.flatten(2)

    # intersection = (pred_flat * gt_flat).sum(dim=2)
    # dice_score = (2 * intersection + smooth) / (pred_flat.sum(dim=2) + gt_flat.sum(dim=2) + smooth)
    # dice_loss = 1 - dice_score.mean()

    # total_loss = loss_weight * (bce_loss + dice_loss)
    # return total_loss



    # Detach:
    # Binary Cross-Entropy Loss (채널별 평균)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1) # (B, 1, H, W) 형식으로 변환
    bce_loss_unweighted = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='mean')

    # Dice Loss 계산 (채널별)
    # Dice Loss를 계산하기 전에 pred_mask에 sigmoid를 적용.
    # F.binary_cross_entropy_with_logits는 내부적으로 sigmoid를 처리하지만, Dice Loss는 확률값(0~1)을 기대.
    pred_mask_sigmoid = pred_mask.sigmoid() # 확률값으로 변환
    pred_flat = pred_mask_sigmoid.flatten(2)  # (B, C, H*W)
    gt_flat = gt_mask.flatten(2)

    intersection = (pred_flat * gt_flat).sum(dim=2)
    dice_score = (2 * intersection + smooth) / (pred_flat.sum(dim=2) + gt_flat.sum(dim=2) + smooth)
    dice_loss_unweighted = 1 - dice_score.mean()

    # 가중치가 적용된 최종 손실
    weighted_bce_loss = loss_weight * bce_loss_unweighted
    weighted_dice_loss = loss_weight * dice_loss_unweighted
    total_loss = weighted_bce_loss + weighted_dice_loss # 또는 loss_weight * (bce_loss_unweighted + dice_loss_unweighted)

    return {
        'loss_mask_total_w': total_loss,           # 최종 역전파에 사용될 가중치 적용된 총 손실
        'loss_mask_bce_uw': bce_loss_unweighted,   # 로깅용: 가중치 미적용 BCE 손실
        'loss_mask_dice_uw': dice_loss_unweighted, # 로깅용: 가중치 미적용 Dice 손실
        'loss_mask_bce_w': weighted_bce_loss,      # 로깅용: 가중치 적용 BCE 손실
        'loss_mask_dice_w': weighted_dice_loss     # 로깅용: 가중치 적용 Dice 손실
    }