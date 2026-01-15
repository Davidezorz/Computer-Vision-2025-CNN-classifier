from parsing.inputParser import parseArgumets

from models.ViT import VisionTransformer
from dataset.dataloader import DatasetManager
from train import train
from utils.utils import getDevice, setupMatplotlib
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import utils
from utils import models_eval 
from omegaconf import OmegaConf
import pydoc





if __name__ == '__main__': 
    flags = parseArgumets()
    print(f"\n flags['config_path']: {flags['config_path']} \n\n\n\n")
    config = OmegaConf.load(flags['config_path'])
    setupMatplotlib()

    save_path = config.get('save_path', 'results/')                             # ◀── folder for saving plots
    model_name = config.model.name                                                # ◀── name of the network
    plot_path = save_path + model_name                                            # ◀── initial name of files will be the same

    np.random.seed(0)                                                           # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed 


    print('getting the device...')                                              #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce fan_in
    print(f'Using device: {device}\n')


    print('model definition...')
    model  = VisionTransformer( patch_size = 4,     # ◀ height/width of patch
                                num_classes = 15,   # ◀ number of output classes
                                in_channels = 1,    # ◀ image channels (3 for RGB)
                                C = 16,            # ◀ embedding dimension
                                H = 4,              # ◀ number of heads
                                N = 3,              # ◀ number of blocks
                                p = 0.1,            # ◀ probability of dropout
                                name=model_name
                            ).to(device)                    
    print(model, "\n")

    n_parameters = utils.utils.numberOfparameters(model)
    print(f"Parameters: {n_parameters}\n")

    
    print('getting dataloaders...')
    folder_path = '.data/'                                                      # ◀─┬ define the folder path
    B = config.training.B                                                       # ◀─┴ and the batch size
    
    transforms_list = utils.utils.processTransforms(config)
    pipe_base, pipe_train, pipe_test, pipe_norm = transforms_list
    dataset_mng = DatasetManager(folder_path, val_split=0.15,                   # ◀─┬ Instantiating the class
                                 pipe_base  = pipe_base,                        #   │ that retrive dataloaders
                                 pipe_train = pipe_train,                       #   │ 
                                 pipe_test  = pipe_test,                        #   │ 
                                 pipe_norm  = pipe_norm)                        #   ╯ 
    
    data_loaders, classes = dataset_mng.get(B)                                  #   ╮ Getting the
    train_loader, val_loader, test_loader = data_loaders                        # ◀─┴ dataloaders
    

    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader:   {len(val_loader)}")
    print(f"test_loader:  {len(test_loader)}\n")
    

     
    print('\ntraining...') 
    if flags['do training']:                                                    # ◀── TRAINING SETUP 
        start_time = time.time()
        optim_class = pydoc.locate(config.training.optim)                       # ◀── define the optimizer class
        optim_opt = OmegaConf.to_container(config.training.opt, resolve=True)
        log_interval = config.training.get('log_interval', 20)
        losses = train(model, train_loader, val_loader,  
                        lr=config.training.lr,                                  # ◀─┬ TRAINING LOOP
                        patience    =config.training.get('patience', 10),       #   │
                        decay_lr    =config.training.get('decay_lr', 1),        #   │
                        log_interval=log_interval,                              #   │
                        use_amp     =config.training.get('use_amp', False),     #   │
                        optim_class =optim_class,                               #   │
                        optim_opt   =optim_opt,                                 #   │
                        device=device, epochs=12*5,                             #   │
                    )                                                           #   ╯ 
        training_time = time.time() - start_time
        print(f"\ntime: {training_time: .3f} s\n")
        

        print('plotting loss...')                                               # ◀─┬ Loss plotting
        r = 1/(len(train_loader) // log_interval)                               #   │
        models_eval.plotLoss(losses['train'], losses['val'],                    #   │
                            title='Loss during training', xlabel='epochs',      #   │
                            ylabel='loss', show=False, ylim=[0, 3.5],           #   │
                            line_at=losses['best_n']-1, r=r,                    #   │
                            save_path=(plot_path + "_train_loss.png"))          #   │
        models_eval.plotLoss(losses['train_accuracy'], losses['val_accuracy'],  #   │
                            title='Accuracy during training', xlabel='epochs',  #   │
                            ylabel='accuracy', show=False, ylim=[0, 1],         #   │
                            line_at=losses['best_n']-1, r=r,                    #   │
                            save_path=(plot_path + "_train_accuracy.png") )     #   ╯
        model.save()
    else:
        model.load()
        training_time = None

    
    print('\ncomputing accuracy...')
    y_true, y_pred = models_eval.getPredictions(model, test_loader, device)       # ◀─┬ Predictions over the test set
    accuracy = models_eval.computeAccuracy(y_true, y_pred )                     # ◀─┬ computing the accuracy
    print(f"accuracy: {accuracy: .4f}\n")                                       #   ╯

    print('compute confusion matrix...')
    cm_name = '_confusion_matrix.png'
    cm = models_eval.computeConfusionMatrix(y_true, y_pred, 
                                            num_classes=len(classes))
    models_eval.plotConfusionMatrix(cm, classes=classes, show=False,
                                    save_path=(plot_path + cm_name))
    

    models_eval.storeExamples(model, train_loader, test_loader, 
                              classes, plot_path, device)


    print('\nsaving log...')
    log_file = save_path + model.name  + ".txt"

    utils.utils.saveLog(
        filepath=log_file,
        model_name=model.name,
        transforms_list=transforms_list,
        config=config,
        n_params=n_parameters,
        accuracy=accuracy,
        training_time=training_time
    )