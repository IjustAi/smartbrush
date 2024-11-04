import torch 
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5  
        self.device = device
        
    def forward(self, predict, target):
        predict = predict.to(self.device)
        target = target.to(self.device)

        intersection = torch.sum(predict * target, dim=(1, 2))  
        union = torch.sum(predict, dim=(1, 2)) + torch.sum(target, dim=(1, 2))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice) 
            
        return dice_loss
