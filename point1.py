from models.parser import convParser
from models.generalCNN import CNN
from dataset.dataloader import DatasetManager
from train import train
from utils.utils import getDevice, setupMatplotlib
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import models_eval 
import argparse










if __name__ == '__main__': # 18

    parser = argparse.ArgumentParser(description="Script for training and plotting")
    parser.add_argument('-train', type=str, default='True', 
                    help='Set to False to skip training')
    args = parser.parse_args()
    do_training = args.train.lower() in ('true', '1', 't', 'yes')
    
    setupMatplotlib()

    np.random.seed(44)                                                          # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed and
    image_dims = (64, 64)                                                       # ◀── image resolution
    config_str = """
                 conv2d    channels: ->8   kernel_size: (3, 3)  stride: (1, 1)
                 relu 
                 maxpool2d                 kernel_size: (2, 2)  stride: (2, 2)
                 conv2d    channels: ->16  kernel_size: (3, 3)  stride: (1, 1)
                 relu 
                 maxpool2d                 kernel_size: (2, 2)  stride: (2, 2)
                 conv2d    channels: ->32  kernel_size: (3, 3)  stride: (1, 1)
                 relu
                 flatten
                 linear    dims:     ->15
                 """
    
    print('parsing...')                                                         #   ╮ Converting the string into 
    parser = convParser()                                                       #   │ a dictionary config using 
    config = parser.str2dict(config_str)                                        # ◀─┴ the convParser class


    print('cnn definition...')                                                  #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce
    print(f'Using device: {device}')

    init_type = torch.nn.init.normal_                                           # ◀─┬ define the initialization class
    init_conf = {'conv': {'std': 0.01}, 'linear': {'std': 0.01}}                # ◀─┴ and its config
    cnn    = CNN(image_dims, config, name='CNN_1', init_type=init_type,         # ◀─┬ Instantiating the Convolutional 
                  init_conf=init_conf).to(device)                               #   ╯ Neural Network
    print(cnn)


    print('getting data...')
    folder_path = '.data/'                                                      # ◀─┬ define the folder path
    B = 32                                                                      # ◀─┴ and the batch size
    dataset_mng = DatasetManager(folder_path, image_dims, val_split=0.15,       # ◀─┬ Instantiating the class
                                 normalize=False)                               #   ╯ that retrive dataloaders

    data_loaders, classes = dataset_mng.get(B)                                  #   ╮ Getting the
    train_loader, val_loader, test_loader = data_loaders                        # ◀─┴ dataloaders

    for x, y in train_loader:                                                   # if debugging needed
        break
    
    """ """
    save_path = 'results/point1/'                                               # folder for saving plots
    if do_training:                                                             # ◀─┬ TRAINING LOOP 
        start = time.time()
        optim_class = torch.optim.SGD                                           # ◀─┬ define the optimizer class
        optim_opt   = {'momentum': 0.9}                                         # ◀─┴ and its config
        losses = train(cnn, train_loader, val_loader, patience=30, lr=11e-4,    # ◀─┬ Training loop  
                    device=device, epochs=12*3,  optim_class=optim_class,       #   │
                    optim_opt=optim_opt, use_amp=True)                          #   ╯ 
        print(f"\ntime: {time.time()-start: .3f} s\n")
        
        print('plotting loss...')                                               # ◀─┬ Loss plotting
        models_eval.plotLoss(losses['train'], losses['val'],                    #   │
                            title='Loss during training', xlabel='steps',       #   │
                            ylabel='loss', show=False,                          #   │
                            save_path=(save_path + "train_loss.png"))           #   │
        models_eval.plotLoss(losses['train_accuracy'], losses['val_accuracy'],  #   │
                            title='Accuracy during training',xlabel='steps',    #   │
                            ylabel='accuracy', show=False,                      #   │
                            save_path=(save_path + "train_accuracy.png") )      #   ╯
        cnn.save()
    else:
        cnn.load()

    print('computing accuracy...')
    y_true, y_pred = models_eval.getPredictions(cnn, test_loader, device)       # ◀─┬ Predictions over the test set
    accuracy = models_eval.computeAccuracy(y_true, y_pred )                     # ◀─┬ computing the accuracy
    print(f"accuracy: {accuracy: .4f}")                                         #   ╯

    print('compute confusion matrix...')
    cm_name = 'confusion_matrix.png'
    cm = models_eval.computeConfusionMatrix(y_true, y_pred, 
                                            num_classes=len(classes))
    models_eval.plotConfusionMatrix(cm, classes=classes, show=False,
                                    save_path=(save_path + cm_name))
    


