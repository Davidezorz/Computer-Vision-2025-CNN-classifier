import torch
from torchvision import datasets, transforms
from datasets import ImageFolder
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

class GetDataloaders:
    
    def __init__(self, folder_path: str, resolution):
        self.folder_path = folder_path
        self.resolution  = resolution
        pass

    def get(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_folder = os.path.join(self.folder_path, 'train')
        test_folder  = os.path.join(self.folder_path, 'test')

        train_dataset = ImageFolder(root=train_folder, transform=transform)
        test_dataset  = ImageFolder(root=test_folder,  transform=transform)




def get_dataloaders(data_dir='.data', batch_size=32, val_split=0.2):
    # 1. Define Transforms
    # Basic transform for now (Resize + ToTensor)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 2. Load the Full Training Data
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Load the entire folder as one dataset first
    full_train_dataset = ImageFolder(root=train_dir, transform=data_transform)
    test_dataset = ImageFolder(root=test_dir, transform=data_transform)

    # 3. Calculate Split Sizes
    # e.g., if you have 1000 images and val_split is 0.2 -> val gets 200, train gets 800
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    # 4. Perform the Split
    # This randomly assigns images to either train_set or val_set
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Fixed seed for reproducibility
    )

    # 5. Extract Class Names
    # Note: access classes from the underlying full dataset, not the subset
    class_names = full_train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}, Test size: {len(test_dataset)}")

    # 6. Create DataLoaders
    # Shuffle Train: YES (Crucial for learning)
    # Shuffle Val/Test: NO (Usually not needed, helps if you want to visualize fixed examples)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names