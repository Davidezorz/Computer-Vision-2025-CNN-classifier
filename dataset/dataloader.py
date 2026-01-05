import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import numpy as np
import itertools




class ChannelMean:                                                              # tensor shape  C H W, but
    def __call__(self, tensor):                                                 # all the channels are 
        return tensor.mean(dim=0, keepdim=True)                                 # equal, then -> 1 H W
    

class DatasetManager:
    
    def __init__(self, folder_path: str, resolution, val_split: float = 0.1, 
                 normalize: bool = True, augmented: list = []):
        self.folder_path = folder_path
        self.resolution  = resolution
        self.val_split   = val_split
        self.normalize   = normalize

        self.base_tranforms = [
            transforms.Resize(self.resolution),
            transforms.ToTensor(),                                              # Converts to [0.0, 1.0]
            ChannelMean()
        ]

        if not self.normalize:
            self.base_tranforms.append(transforms.Lambda(lambda x: x * 255.0))

        self.augmented_list = augmented


    def get(self, B: int = 32):
        train_folder = os.path.join(self.folder_path, 'train')                  # ◀─┬ get the training and 
        test_folder  = os.path.join(self.folder_path, 'test')                   # ◀─┴ test folders

        base_tranforms = transforms.Compose(self.base_tranforms)
        train_data = ImageFolder(root=train_folder, transform=base_tranforms)   # ◀─┬ open data and apply
        test_data  = ImageFolder(root=test_folder,  transform=base_tranforms)   # ◀─╯ transformations

        targets = train_data.targets                                            # ◀── Extract the labels 
        train_idx, val_idx = train_test_split(                                  # ◀─┬ Use sk to generate stratified indices
            np.arange(len(targets)),                                            #   │ ◀ array of indices to split
            test_size=self.val_split,                                           #   │ 
            shuffle=True,                                                       #   │ ◀ shuffle
            stratify=targets                                                    #   │ ◀ stratify by label
        )                                                                       #  ─╯
        
        train_subsets = []                                                      # ◀─┬ Apply data agumentation if needed
        for r in range(len(self.augmented_list) + 1):                           #   │ by combining the basics transformations               
            for combo in itertools.combinations(self.augmented_list, r):        #   │ with all combinations of the 
                current_transforms = self.base_tranforms + list(combo)          #   │ transformations
                current_transforms = transforms.Compose(current_transforms)     #   │ ◀ Base Steps + Current Combination
                dataset = ImageFolder(root=train_folder,                        #   │ ◀  Load a fresh dataset with 
                                      transform=current_transforms)             #   │    that specific transforms
                subset  = Subset(dataset, train_idx)                            #   │ ◀ Apply the SAME training indices 
                train_subsets.append(subset)                                    #  ─╯   to this new version of the data
        
        X_train = ConcatDataset(train_subsets)                                  # ◀─┬ Create the datasets
        X_val   = Subset(train_data, val_idx)                                   #  ─╯ 

        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val, batch_size=B, shuffle=False)           #   │ for each data subdset
        test_loader  = DataLoader(test_data, batch_size=B, shuffle=False)       # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
