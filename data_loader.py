import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from different_level_mask import GaussianBlur

class MaskedImageDataset(Dataset):
    def __init__(self, original_image_path, mask_image_path, levels, text_label, batch_size, image_size, device):
        self.original_image_path = original_image_path
        self.mask_image_path = mask_image_path
        self.levels = levels
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        inputs = self.processor(text=text_label, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            self.text_embeddings = self.model.get_text_features(**inputs).squeeze(0)  

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)), 
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        self.s_set = []
        
        for level in self.levels:
            mask_s = GaussianBlur(mask_image_path, level, 0)
            mask_s_array = np.array(mask_s).astype(np.float32) / 255.0  
            self.s_set.append(mask_s_array)

        x0 = Image.open(self.original_image_path).resize((self.image_size, self.image_size))
        x0_array = np.array(x0).astype(np.float32) / 255.0  
        self.x0 = torch.tensor(x0_array).permute(2, 0, 1).to(self.device)
        
        mask_image = Image.open(self.mask_image_path).resize((self.image_size, self.image_size))
        self.mask = torch.tensor(np.array(mask_image).astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)

        to_gray = transforms.Grayscale(num_output_channels=1)
        self.mask = to_gray(self.mask)
        self.mask = (torch.sigmoid(self.mask) > 0.5).float()

    def __len__(self):
        return len(self.s_set)

    def __getitem__(self, idx):
        mask_array = self.s_set[idx]
        train_data = self.transform(Image.fromarray((mask_array * 255).astype(np.uint8))) 
        train_data = (train_data > 0.5).float()   
        label = self.text_embeddings.to(self.device) 

        return train_data, label, self.x0, self.mask
