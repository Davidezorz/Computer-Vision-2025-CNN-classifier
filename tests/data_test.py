from dataset.dataloader import *

if __name__ == '__main__':
    resolution=(28, 28)
    dataset_mng = DatasetManager(folder_path=".data/",
                                 resolution=resolution,
                                 val_split=0.15)
    
    data_loaders, classes = dataset_mng.get()
    train_loader, val_loader, test_loader = data_loaders

    print(f"Classes: ")

    for cls in classes:
        print(f"  - {cls}")
    print(f"Train size: {len(train_loader)}, Val size: {len(val_loader)}, Test size: {len(test_loader)} \n")



    for x, y in train_loader:
        print("x.shape: ", x.shape, "\n")
        print("max: ", x[0].max())
        print("min: ", x[0].min())
        print("avg: ", x[0].mean())
        
        print('portion of x: ')
        print(x[0, :resolution[0]//4, :resolution[1]//4 ])
        print('y: ')
        print(y[0])
        break
