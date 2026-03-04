import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy.ndimage import zoom

class UterineDataset(Dataset):
    def __init__(self, csv_path, target_shape=(128, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape

    def __len__(self):
        return len(self.df)

    def get_bbox_from_contour(self, image, mask):
        """
        Takes a contour mask and finds the boundary to crop the image.
        """
        # Find all (Z, Y, X) coordinates where the contour exists
        coords = np.argwhere(mask > 0)
        
        if coords.size == 0:
            return image # Fallback if mask is empty
            
        # Extract the min and max of each axis to form a 'virtual' bounding box
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Add 10% padding so the model sees the 'edges' of the uterus
        z_pad = int((z_max - z_min) * 0.1)
        y_pad = int((y_max - y_min) * 0.1)
        x_pad = int((x_max - x_min) * 0.1)

        # Apply crop with safety boundaries
        img_crop = image[
            max(0, z_min-z_pad) : min(image.shape[0], z_max+z_pad),
            max(0, y_min-y_pad) : min(image.shape[1], y_max+y_pad),
            max(0, x_min-x_pad) : min(image.shape[2], x_max+x_pad)
        ]
        return img_crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img = nib.load(row['image_path']).get_fdata().astype(np.float32)
        
        # 2. Use Contour Mask for Cropping (if available)
        mask_path = row['mask_path']
        if pd.notna(mask_path) and os.path.exists(str(mask_path)):
            mask = nib.load(mask_path).get_fdata()
            img = self.get_bbox_from_contour(img, mask)
        
        # 3. Resize to 128x128x128
        # Using zoom for 3D interpolation
        factors = [t/s for t, s in zip(self.target_shape, img.shape)]
        img = zoom(img, factors, order=1)
        
        # 4. Normalization
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        
        # 5. Format for PyTorch (1, D, H, W)
        return torch.FloatTensor(img).unsqueeze(0), torch.tensor(int(row['label']))