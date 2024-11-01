import torch
import torch.nn as nn
import numpy as np 
from utilities import *
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=512, height=28):  
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_mask = ResidualConvBlock(1, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat) 
        
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),                    
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        
        self.out_mask = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, 1, 3, 1, 1),  
        )

    def forward(self, x, t, c=None, mask=None):
        """
        x : (batch, in_channels, h, w) : input image
        t : (batch, n_cfeat)            : time step
        c : (batch, n_classes)          : context label
        mask: (batch, 1, h, w)          : input mask
        
        """

        x = self.init_conv(x)
        mask = self.init_mask(mask)
        down_mask1 = self.down1(mask)
        down_mask2 = self.down2(down_mask1)
        hidden_mask_vec = self.to_vec(down_mask2)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden_image_vec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hidden_image_vec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out_image = self.out(torch.cat((up3, x), 1))
        
        # need to debug this part of code 
        up_mask1 = self.up0(hidden_mask_vec)
        up_mask2 = self.up1(cemb1 * up_mask1+temb1, down_mask2)
        up_mask3 = self.up2(cemb2*up_mask2+temb2, down_mask1)
        out_mask = self.out_mask(torch.cat((up_mask3, mask), 1))

        return out_image, out_mask