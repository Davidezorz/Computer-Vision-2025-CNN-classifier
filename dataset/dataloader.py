import torch
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import numpy as np

    


class DatasetManager:
    """ 
    This class helps load images from a folder 'folder_path', expected to have 
    two sub-folders inside: 
    1. 'train' (for training)
    2. 'test' (for testing)

    What this class does:
    1. It loads images from the 'train' folder.
    2. It automatically splits this data into a Training set and a Validation set.
    3. It applies image transformations (like resizing or data augmentation).
       - It is smart: it applies 'pipe_train' (augmentation) only to the 
         training split.
       - It applies 'pipe_test' (clean) to the validation and test splits.
    4. It gives you ready-to-use PyTorch DataLoaders and the classes names.

    Args:
        folder_path (str):    The main folder containing 'train' and 'test' folders.
        val_split   (float):  The percentage of training data to use for validation.
        pipe_base   (list):   Transforms applied for ALL images.
        pipe_train  (list):   Transforms ONLY for training.
        pipe_test   (list):   Transforms ONLY for validation/test.
        pipe_norm   (list):   Normalization values, applied at the very end.
Returns:
        data_loaders (list): A list containing [train_loader, val_loader, test_loader]
            - train_loader: torch DataLoader containing training images divided in batches.
            - val_loader:   torch DataLoader containing validation images divided in batches.
            - test_loader:  torch DataLoader containing test images divided in batches.
        classes (list):
            - The list of class names (labels) found in the train folder.
    """

    def __init__(self, folder_path: str, val_split: float = 0.1,
                 pipe_base:  list = [], 
                 pipe_train: list = [],
                 pipe_test:  list = [],
                 pipe_norm:  list = []
                ):
        
        self.folder_path = folder_path
        self.val_split   = val_split

        self.pipe_base  = pipe_base  if pipe_base else []
        self.pipe_train = pipe_train if pipe_train else []
        self.pipe_test  = pipe_test  if pipe_test else []
        self.toTensor   = [transforms.ToTensor()]
        self.pipe_norm  = pipe_norm if pipe_norm else []


    def get(self, B: int = 32):
        train_folder = os.path.join(self.folder_path, 'train')                  # ◀─┬ get the training and 
        test_folder  = os.path.join(self.folder_path, 'test')                   # ◀─┴ test folders

        tranforms_train = Compose(self.pipe_base + self.pipe_train +            # ◀─┬ Apply data agumentation if needed:
                                  self.toTensor + self.pipe_norm)               #   ╯ base tranforms ▶ augument ▶ tensor ▶ normalize
        tranforms_test  = Compose(self.pipe_base + self.pipe_test +             # ◀─┬ Apply test and validation transofrms:
                                  self.toTensor + self.pipe_norm)               #   ╯ test tranforms ▶ tensor ▶ normalize
        
        print(f'tranforms_train:    \n{tranforms_train}')
        print('\n')
        print(f'tranforms_test:     \n{tranforms_test}')
        print('\n')

        train_data = ImageFolder(root=train_folder, transform=tranforms_train)  # ◀─┬ open data and apply
        val_data   = ImageFolder(root=train_folder, transform=tranforms_test)   # ◀─┤ transformations
        X_test     = ImageFolder(root=test_folder,  transform=tranforms_test)   # ◀─╯ 
        
        targets = train_data.targets                                            # ◀── Extract the labels 
        train_idx, val_idx = train_test_split(                                  # ◀─┬ Use sk to generate stratified indices
            np.arange(len(targets)),                                            #   │ ◀ array of indices to split
            test_size=self.val_split,                                           #   │ 
            shuffle=True,                                                       #   │ ◀ shuffle
            stratify=targets                                                    #   │ ◀ stratify by label
        )                                                                       #  ─╯

        X_train = Subset(train_data, train_idx)
        X_val   = Subset(val_data, val_idx)
        
        train_loader = DataLoader(X_train, batch_size=B, shuffle=True)          # ◀─╮ crete the Dataloader  
        val_loader   = DataLoader(X_val,   batch_size=B, shuffle=False)         #   │ for each data subdset
        test_loader  = DataLoader(X_test,  batch_size=B, shuffle=False)         # ◀─╯
        data_loaders = [train_loader, val_loader, test_loader]                  # ◀── pack the Dataloaders

        return data_loaders, train_data.classes
