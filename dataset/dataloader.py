import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os




class ChannelMean:                                                              # tensor shape  C H W, but
    def __call__(self, tensor):                                                 # all the channels are 
        return tensor.mean(dim=0, keepdim=True)                                 # equal, then -> 1 H W
    

class DatasetManager:
    
    def __init__(self, folder_path: str, resolution, val_split: float = 0.1, 
                 normalize: bool = True):
        self.folder_path = folder_path
        self.resolution  = resolution
        self.val_split   = val_split
        self.normalize   = normalize

        transform_steps = [
            transforms.Resize(self.resolution),
            transforms.ToTensor(),  # This always converts to [0.0, 1.0]
        ]

        if not self.normalize:
            transform_steps.append(transforms.Lambda(lambda x: x * 255.0))
        transform_steps.append(ChannelMean())

        self.transform = transforms.Compose(transform_steps)


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
            #generator=torch.Generator().manual_seed(42)                         # ◀── Seed for reproducibility
        )

        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val, batch_size=B, shuffle=False)           #   │ for each data subdset
        test_loader  = DataLoader(test_data, batch_size=B, shuffle=False)       # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
