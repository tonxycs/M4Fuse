import torch
import torch.nn as nn
import torch.nn.functional as F

class BraTSLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.7, 
                 class_dice_weights=[1.0, 2.0, 3.0, 5.0],  
                 device='cuda'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.class_dice_weights = torch.tensor(class_dice_weights, device=device)
        self.device = device
        self.eps = 1e-6  

    def forward(self, pred, target):

        ce = self.ce_loss(pred, target)
        

        pred_softmax = F.softmax(pred, dim=1)  # (B, C, D, H, W)
        num_classes = pred.shape[1]
        dice = 0.0
        
        for c in range(num_classes):
            pred_mask = pred_softmax[:, c, ...]
            target_mask = (target == c).float()
            
            intersection = (pred_mask * target_mask).sum() + self.eps
            union = pred_mask.sum() + target_mask.sum() + self.eps
            class_dice = 1 - 2.0 * intersection / union
            dice += class_dice * self.class_dice_weights[c]
        
        dice /= self.class_dice_weights.sum()
        

        total_loss = (1 - self.dice_weight) * ce + self.dice_weight * dice
        return total_loss
