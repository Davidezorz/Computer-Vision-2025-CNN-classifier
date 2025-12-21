from dataset.dataloader import *

if __name__ == '__main__':
    dataset_mng = DatasetManager(folder_path=".data/",
                                 resolution=(28, 28),
                                 val_split=0.15)
    
    data_loaders, classes = dataset_mng.get()
    train_loader, val_loader, test_loader = data_loaders

    print(f"Classes: ")

    for cls in classes:
        print(f"  - {cls}")
    print(f"Train size: {len(train_loader)}, Val size: {len(val_loader)}, Test size: {len(test_loader)}")
