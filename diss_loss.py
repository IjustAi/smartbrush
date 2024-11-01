import torch 
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self,device):
        super(DiceLoss, self).__init__()  
        self.smooth = 1e-5 #prevent 0 division
        self.device = device
        
    def forward(self,predict,target):
        predict = predict.to(self.device)
        target = target.to(self.device)

        target_onehot = torch.zeros(predict.size(), device=self.device)
        target_onehot.scatter_(1, target.long(), 1) 

        intersection = torch.sum(predict*target_onehot,dim=2)
        union = torch.sum(predict.pow(2),dim=2) +torch.sum(target_onehot,dim=2)

        dice = (2*intersection+self.smooth) /(union+self.smooth)
        dice_loss= 1-torch.mean(dice)
            
        return dice_loss