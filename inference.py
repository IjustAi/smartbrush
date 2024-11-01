import torch
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from utilities import *

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
timesteps = 500
n_feat = 64
n_cfeat = 5
height = 16
beta1 = 1e-4
beta2 = 0.02
batch_size = 100
n_epoch = 32

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device, dtype=torch.float32) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp().to(device)  
ab_t[0] = 1 

text='monkey'
save_dir = '/Users/hexu/Documents/NTU-Learn/deep learning/project/smartBrush_new/checkpoint'

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_embeddings = model.get_text_features(**inputs).squeeze(0).to(device)  

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm_context(n_sample, mask,x0, save_rate=20):
    #原始数据
    samples = torch.randn(n_sample, 3, height, height).to(device)  
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        #每个时间步的噪声
        z = torch.randn_like(samples) if i > 1 else 0

        noise_pred,mask_pred = model(x=samples, t=t, c=text_embeddings, mask=mask) 
        mask = (torch.sigmoid(mask_pred)>0.5).float()

        samples = noise_pred * mask + x0 * (1-mask)
        
        #每个时间步对原始数据进行降噪
        samples = denoise_add_noise(samples, i, noise_pred, z)

        
        if i % save_rate==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()
    
def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani
model = Unet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
model.load_state_dict(torch.load(f"{save_dir}/model_epoch_100.pth", map_location=device))
model.eval()



ctx = torch.tensor([
    # Hero: 
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
    [1,0,0,0,0],  [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0],
]).float().to(device)
samples, intermediate = sample_ddpm_context(ctx.shape[0], ctx)
animation_ddpm_context = plot_sample(intermediate,ctx.shape[0],8,save_dir, "ani_run", None,save=False)
HTML(animation_ddpm_context.to_jshtml())
