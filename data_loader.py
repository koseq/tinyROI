import os

import cv2
import numpy as np

import torch
from torchvision import transforms as T


    
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self,img_name_list,roi_inf_size):
        self.image_name_list = img_name_list
        
        self.roi_transform = T.Compose([
            T.ToTensor(),
            T.Resize((roi_inf_size, roi_inf_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.det_transform = T.Compose([
            T.ToTensor(),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):
        image = cv2.imread(self.image_name_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]
        
        image_roi = self.roi_transform(image)
        image_det = self.det_transform(image)

        sample = {
            'imidx':torch.tensor(idx), 
            'image_roi':image_roi.to(torch.float),
            'image_det':image_det.to(torch.float),
            'height': torch.tensor(H),
            'width': torch.tensor(W),
            'fname': os.path.basename(self.image_name_list[idx])
        }            
    
        return sample