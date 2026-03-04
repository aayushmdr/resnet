import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import os
import gc # Added for memory management

class UterineDataset(Dataset):
    def __init__(self, csv_path, target_shape=(128, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape

    def __len__(self):
        return len(self.df)

    def get_bbox_crop(self, image, mask):
        coords = np.argwhere(mask > 0)
        if coords.size == 0: return image
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        # 5-pixel buffer
        return image[max(0, z_min-5):z_max+5, max(0, y_min-5):y_max+5, max(0, x_min-5):x_max+5]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load NIfTI (ensure path is absolute for SSH consistency)
        img_obj = nib.load(row['image_path'])
        img_data = img_obj.get_fdata().astype(np.float32)
        
        if pd.notna(row['mask_path']) and os.path.exists(str(row['mask_path'])):
            mask_data = nib.load(row['mask_path']).get_fdata()
            img_data = self.get_bbox_crop(img_data, mask_data)
            del mask_data # Free memory immediately
        
        # Resize
        zoom_factors = [t / s for t, s in zip(self.target_shape, img_data.shape)]
        img_data = zoom(img_data, zoom_factors, order=1)
        
        # Normalize
        img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-8)
        
        # Explicitly trigger garbage collection for the large raw objects
        gc.collect() 
        
        return torch.FloatTensor(img_data).unsqueeze(0), torch.tensor(int(row['label']))