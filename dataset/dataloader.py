import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

"""
─  │
┌  ┐
└  ┘
┬  ┴  ├  ┤
┼
"""


"""
.data/
  ├ train
  │  ├ Bedroom/
  │  │  ├ image_0001.jpg
  │  │  ├ ...
  │  ├ InsideCity/
  │  │  ├ image_0005.jpg
  │  │  ├ ...
  │  ├ OpenCountry/
  │  │  ├ image_00012.jpg
  │  │  ├ ...
  │  ├ Coast/
  │  │  ├ image_0004.jpg
  │  │  ├ ...
  │  ├ Kitchen/
  │  │  ├ image_0002.jpg
  │  │  ├ ...
  │  ├ Store/
  │  │  ├ image_0003.jpg
  │  │  ├ ...
  │  ├ Forest/
  │  │  ├ image_0002.jpg
  │  │  ├ ...
  │  ├ LivingRoom/
  │  │  ├ image_0009.jpg
  │  │  ├ ...
  │  ├ Street/
  │  │  ├ image_0008.jpg
  │  │  ├ ...
  │  ├ Highway/
  │  │  ├ image_0015.jpg
  │  │  ├ ...
  │  ├ Mountain/
  │  │  ├ image_0019.jpg
  │  │  ├ ...
  │  ├ Suburb/
  │  │  ├ image_0011.jpg
  │  │  ├ ...
  │  ├ Industrial/
  │  │  ├ image_0021.jpg
  │  │  ├ ...
  │  ├ Office/
  │  │  ├ image_0006.jpg
  │  │  ├ ...
  │  └ TallBuilding/
  │     ├ image_0012.jpg
  │     └ ...
  └ test/
    ├ ... 
    │  ├ ... 
    │  └ ...
    ├ ... 
    ...
"""

class DatasetManager:
    
    def __init__(self, folder_path: str, resolution, val_split: int = 0.1):
        self.folder_path = folder_path
        self.resolution  = resolution
        self.val_split   = val_split

        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
        ])



    def get(self, B: int = 32):
        train_folder = os.path.join(self.folder_path, 'train')                  # ◀─┬ get the training and 
        test_folder  = os.path.join(self.folder_path, 'test')                   # ◀─┴ test folders

        train_data = ImageFolder(root=train_folder, transform=self.transform)   # ◀─┬ open data and apply
        test_data  = ImageFolder(root=test_folder,  transform=self.transform)   # ◀─┴ transformations
        
        val_size   = int(len(train_data) * self.val_split)                      # ◀─╮ compute the actual training 
        train_size = len(train_data) - val_size                                 # ◀─╯ and validation dims

        X_train, X_val = random_split(                                          # ◀─╮ 
            train_data,                                                         #   │ Apply the split according 
            [train_size, val_size],                                             #   ╰ to the computed dimensions
            generator=torch.Generator().manual_seed(42)                         # ◀── Seed for reproducibility
        )

        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val, batch_size=B, shuffle=False)           #   │ for each data subdset
        test_loader  = DataLoader(test_data, batch_size=B, shuffle=False)       # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
