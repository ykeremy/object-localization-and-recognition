import os

import torch
from torch.utils.data import Dataset
from skimage import io


class ImageNetDataset(Dataset):
    """Small ImageNet Dataset"""

    mapping = {
        'n01615121': 0,  # Bird
        'n02099601': 1,  # Dog
        'n02123159': 2,  # Cat
        'n02129604': 3,  # Tiger
        'n02317335': 4,  # Starfish
        'n02391049': 5,  # Zebra
        'n02410509': 6,  # Bison
        'n02422699': 7,  # Antelope
        'n02481823': 8,  # Chimpanzee
        'n02504458': 9,  # Elephant
    }

    def __init__(self, data_dir, transform=None):
        self.dir = data_dir
        self.transform = transform

        self.filenames = []
        subfolders = os.listdir(self.dir)
        for subfolder in subfolders:
            img_path_list = os.listdir(os.path.join(self.dir, subfolder))
            img_path_list = [os.path.join(self.dir, subfolder, path)
                             for path in img_path_list]
            self.filenames.extend(img_path_list)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
