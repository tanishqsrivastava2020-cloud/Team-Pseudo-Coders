import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (256,256))
        image = image.transpose(2,0,1)
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256,256))
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
