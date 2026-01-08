import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import numpy as np
import itertools

    


class DatasetManager:

    def __init__(self, folder_path: str, resolution, val_split: float = 0.1,
                 pipe_base: list = [], 
                 pipes_aug: list = [],
                 pipe_norm: list = []
                ):
        
        self.folder_path = folder_path
        self.resolution  = resolution
        self.val_split   = val_split

        resize =  transforms.Resize(self.resolution)
        self.pipe_base  = pipe_base if pipe_base else [resize]
        self.toTensor   = [transforms.ToTensor()]
        self.pipes_aug  = pipes_aug if pipes_aug else []
        self.pipe_norm  = pipe_norm if pipe_norm else []


    def get(self, B: int = 32):
        train_folder = os.path.join(self.folder_path, 'train')                  # ◀─┬ get the training and 
        test_folder  = os.path.join(self.folder_path, 'test')                   # ◀─┴ test folders

        base_tranforms = transforms.Compose(self.pipe_base + self.toTensor +
                                            self.pipe_norm)
        train_data = ImageFolder(root=train_folder, transform=base_tranforms)   # ◀─┬ open data and apply
        test_data  = ImageFolder(root=test_folder,  transform=base_tranforms)   # ◀─╯ transformations
        
        targets = train_data.targets                                            # ◀── Extract the labels 
        train_idx, val_idx = train_test_split(                                  # ◀─┬ Use sk to generate stratified indices
            np.arange(len(targets)),                                            #   │ ◀ array of indices to split
            test_size=self.val_split,                                           #   │ 
            shuffle=True,                                                       #   │ ◀ shuffle
            stratify=targets                                                    #   │ ◀ stratify by label
        )                                                                       #  ─╯
        
        
        subset_original = Subset(train_data, train_idx)                         # Original Data (Base Transform)
        train_subsets = [subset_original]
        
        for pipe_aug in self.pipes_aug:                                         # ◀─┬ Apply data agumentation if present
            aug_transf = transforms.Compose(self.pipe_base + pipe_aug +         #   │ ◀ compute local transformation:
                                            self.toTensor + self.pipe_norm)     #   │   resize ▶ augument ▶ tensor ▶ normalize

            aug_dataset = ImageFolder(root=train_folder, transform=aug_transf)  #   │ ◀ Create a fresh view of the data with this transform            
            aug_subset  = Subset(aug_dataset, train_idx)                        #   │ ◀ Apply train indices (so we augment only
            train_subsets.append(aug_subset)                                    #  ─╯   the training images)
        
        X_train = ConcatDataset(train_subsets)                                  # ◀─┬ Create the datasets
        X_val   = Subset(train_data, val_idx)                                   #  ─╯ 

        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val, batch_size=B, shuffle=False)           #   │ for each data subdset
        test_loader  = DataLoader(test_data, batch_size=B, shuffle=False)       # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
