import torch
import os 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from data_loader import MaskedImageDataset
from different_level_mask import GaussianBlur
from diss_loss import DiceLoss
from u_net import Unet
from utilities import *

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    timesteps  = 500
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp().to(device)  
    ab_t[0] = 1
    beta1 = 1e-4
    beta2 = 0.02
    n_feat = 128
    batch_size = 1
    epochs = 101
    lrate=1e-3
    gamma = 0.01
    image_size=32 # no computing power for 256 need high computing power 
    height =image_size

    checkpoint_dir ='./checkpoint'
    model = Unet(in_channels=3, n_feat=n_feat, n_cfeat=512, height=height).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    diss_loss=DiceLoss(device)

    levels = [41, 81, 121, 161, 201, 241, 281, 321, 361,401]
    folder_path_1 = '/Users/chenyufeng/desktop/segtrackv2/JPEGImages'
    folder_path_2 = '/Users/chenyufeng/desktop/segtrackv2/GroundTruth'
    dataset = []

    for subfolder in os.listdir(folder_path_1):
        label = subfolder
        subfolder_path_1 = os.path.join(folder_path_1, subfolder)
        if os.path.isdir(subfolder_path_1):
            for file_name in os.listdir(subfolder_path_1):
                file_path_1 = os.path.join(subfolder_path_1, file_name)  
                file_basename_1 = get_file_basename(file_path_1)

                subfolder_path_2 = os.path.join(folder_path_2, subfolder)

                if os.path.exists(subfolder_path_2):
                    ground_truth_files = get_all_files_recursive(subfolder_path_2)
                    for ground_truth_file_path in ground_truth_files:
                        file_basename_2 = get_file_basename(ground_truth_file_path)
                        if file_basename_2 == file_basename_1:
                            sub_dataset = MaskedImageDataset(file_path_1, ground_truth_file_path, levels, label, batch_size, image_size, device)
                            dataset.append(sub_dataset)
                else:
                    print(f"GroundTruth subfolder {subfolder_path_2} does not exist.")
   
    dataset = ConcatDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model.train()
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        optim.param_groups[0]['lr'] = lrate * (1 - epoch / epochs)
        
        pbar = tqdm(dataloader, mininterval=2)
        epoch_loss = 0  
        '''
        x:blured mask 
        c:text 
        x0:original image
        mask:perfect mask 
        x->model->y  diceloss(y,mask)
        '''
        for x, c, x0,mask in pbar:
            
            optim.zero_grad()
            x = x.to(device)
            c = c.to(x)

            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)

            noise = torch.randn_like(x0)

            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)

            xt = perturb_input(x0, t, noise,ab_t)
            xt = xt * mask + x0 * (1 - mask)

            pred_noise, pred_mask = model(xt, t/ timesteps , c, x)
            pred_mask = (torch.sigmoid(pred_mask)>0.5).float()
            d_loss = diss_loss(pred_mask, mask)
            f_loss = F.mse_loss(pred_noise, noise)
    
            loss = d_loss*0.01 + f_loss
            #print(f'mask_loss = {d_loss}, ordinary diffusion loss ={f_loss}')
            loss.backward()
            optim.step()

            epoch_loss += loss.item() 
        avg_loss = epoch_loss /len(dataloader) 

        print(f"Epoch {epoch+1} finished with loss {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optim, epoch + 1,avg_loss, checkpoint_path)

if __name__ == '__main__':
    main()