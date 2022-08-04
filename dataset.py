from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class LowNormalDataset(Dataset):
    def __init__(self, root_normal, root_low, transform=None):
        self.root_normal = root_normal
        self.root_low = root_low
        self.transform = transform

        self.normal_images = os.listdir(root_normal)
        self.low_images = os.listdir(root_low)
        self.length_dataset = max(len(self.normal_images), len(self.low_images))
        self.normal_len = len(self.normal_images)
        self.low_len = len(self.low_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        normal_img = self.normal_images[index % self.normal_len]
        low_img = self.low_images[index % self.low_len]

        normal_path = os.path.join(self.root_normal, normal_img)
        low_path = os.path.join(self.root_low, low_img)

        normal_img = np.array(Image.open(normal_path).convert("RGB"))
        low_img = np.array(Image.open(low_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=normal_img, image0=low_img)
            normal_img = augmentations["image"]
            low_img = augmentations["image0"]

        return normal_img, low_img
