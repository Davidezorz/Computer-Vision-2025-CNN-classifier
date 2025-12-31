import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import numpy as np




class ChannelMean:                                                              # tensor shape  C H W, but
    def __call__(self, tensor):                                                 # all the channels are 
        return tensor.mean(dim=0, keepdim=True)                                 # equal, then -> 1 H W
    

class DatasetManager:
    
    def __init__(self, folder_path: str, resolution, val_split: float = 0.1, 
                 normalize: bool = True, agumented: list = []):
        self.folder_path = folder_path
        self.resolution  = resolution
        self.val_split   = val_split
        self.normalize   = normalize

        transform_steps = [
            transforms.Resize(self.resolution),
            transforms.ToTensor(),                                              # Converts to [0.0, 1.0]
            ChannelMean()
        ]

        if not self.normalize:
            transform_steps.append(transforms.Lambda(lambda x: x * 255.0))

        self.agumented = agumented != []
        self.transform_agu = transforms.Compose(transform_steps + agumented)
        self.transform = transforms.Compose(transform_steps)


    def get(self, B: int = 32):
        train_folder = os.path.join(self.folder_path, 'train')                  # ◀─┬ get the training and 
        test_folder  = os.path.join(self.folder_path, 'test')                   # ◀─┴ test folders

        train_data = ImageFolder(root=train_folder, transform=self.transform)   # ◀─┬ open data and apply
        test_data  = ImageFolder(root=test_folder,  transform=self.transform)   # ◀─╯ transformations

        targets = train_data.targets                                            # ◀── Extract the labels 
        train_idx, val_idx = train_test_split(                                  # ◀─┬ Use sk to generate stratified indices
            np.arange(len(targets)),                                            #   │ ◀ array of indices to split
            test_size=self.val_split,                                           #   │ 
            shuffle=True,                                                       #   │ ◀ shuffle
            stratify=targets                                                    #   │ ◀ stratify by label
        )                                                                       #  ─╯

        X_train = Subset(train_data, train_idx)                                 #  ─╮ create the Subsets 
        X_val   = Subset(train_data, val_idx)                                   #  ─╯ using the indices

        if self.agumented:                                                      # ◀─┬ If we have to apply data agumentation
            agu_data    = ImageFolder(root=train_folder,                        #   │ ◀ apply agumented transfomration
                                      transform=self.transform_agu)             #   │
            X_train_aug = Subset(agu_data, train_idx)                           #   │ ◀ get the train subset
            X_train     = ConcatDataset([X_train, X_train_aug])                 #  ─╯ ◀ update X_train
    

        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val, batch_size=B, shuffle=False)           #   │ for each data subdset
        test_loader  = DataLoader(test_data, batch_size=B, shuffle=False)       # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
