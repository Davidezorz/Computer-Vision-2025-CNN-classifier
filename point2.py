from models.parser import convParser
from models.generalCNN import CNN
from dataset.dataloader import DatasetManager
from train import train
from utils.utils import getDevice, setupMatplotlib, parseArgumets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import utils
from utils import models_eval 
from torchvision import transforms
from omegaconf import OmegaConf
import pydoc
import datetime



def save_experiment_log(filepath, model_name, config, 
                        n_params, accuracy, training_time):
    """ Appends a summary of the experiment to a text file. """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 50
    
    with open(filepath, "w") as f:
        f.write(f"\n{separator}\n")
        f.write(f"EXPERIMENT LOG: {timestamp}\n")
        f.write(f"{separator}\n")
        
        f.write(f"Model Name:      {model_name}\n")
        f.write(f"Parameters:      {n_params:,}\n") 
        f.write(f"Learning Rate:   {config.training.lr}\n")
        f.write(f"Batch Size:      {32}\n") 
        f.write(f"Optimizer:       {config.training.optim}\n")
        f.write(f"Transforms:      {config.data.transforms}\n") 
        
        f.write(f"Test Accuracy:   {accuracy:.4f}\n")

        training_time = f"{training_time: .4f}" if training_time else '-'
        f.write(f"training time:   {training_time}\n")
        f.write(f"{separator}\n\n")

    print(f"Log saved to {filepath}")



def processTransforms(config):
    real_transforms = []
    for item in config.data.transforms:
        cls = pydoc.locate(item.type)
        params = item.params if item.params else {}
        obj = cls(**params)
        real_transforms.append(obj)
    return real_transforms










if __name__ == '__main__': # 18
    flags = parseArgumets()
    config = OmegaConf.load(flags['config_path'])
    setupMatplotlib()

    np.random.seed(18)                                                          # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed 

    image_dims = (64, 64)                                                       # ◀── Setting image resolution
    
    print('parsing...')                                                         #   ╮ Converting the string into 
    parser = convParser()                                                       #   │ a dictionary config using 
    nn_config = parser.str2dict(config.model.nn_config)                         # ◀─┴ the convParser class


    print('cnn definition...')                                                  #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce fan_in
    print(f'Using device: {device}')


    init_type = pydoc.locate(config.model.init.type)                            # ◀─┬ define the initialization class
    init_conf = OmegaConf.to_container(config.model.init.conf, resolve=True)    # ◀─┴ and its config
    name = config.model.name
    cnn  = CNN(image_dims, nn_config, name=name, init_type=init_type,           # ◀─┬ Instantiating the Convolutional 
                  init_conf=init_conf).to(device)                               #   ╯ Neural Network
    print(cnn)

    n_parameters = utils.utils.numberOfparameters(cnn)
    print(f"Parameters: {n_parameters}\n")

    
    print('getting data...')
    folder_path = '.data/'                                                      # ◀─┬ define the folder path
    B = config.training.B                                                       # ◀─┴ and the batch size
    
    normalize = config.data.normalize
    tranformations = processTransforms(config)
    dataset_mng = DatasetManager(folder_path, image_dims, val_split=0.15,       # ◀─┬ Instantiating the class
                                 augmented=tranformations,                      #   │ that retrive dataloaders
                                 normalize=normalize)                           #   ╯ 
    
    data_loaders, classes = dataset_mng.get(B)                                  #   ╮ Getting the
    train_loader, val_loader, test_loader = data_loaders                        # ◀─┴ dataloaders

    print(f"train_loader: {len(train_loader)}")
    print(f"train_loader: {len(val_loader)}")
    print(f"val_loader: {len(val_loader)}")

    save_path = 'results/point2/'                                               # folder for saving plots
    plot_path = save_path + cnn.name 

    if flags['do training']:                                                    # ◀── TRAINING LOOP 
        start_time = time.time()
        optim_class = pydoc.locate(config.training.optim)                       # ◀── define the optimizer class
        optim_opt = OmegaConf.to_container(config.training.opt, resolve=True)
        lr = config.training.lr
        decay_lr = config.training.decay_lr
        log_interval = config.training.log_interval
        losses = train(cnn, train_loader, val_loader, patience=30, lr=lr,       # ◀─┬ Training loop  
                    device=device, epochs=12*3,  optim_class=optim_class,       #   │
                    optim_opt=optim_opt, decay_lr=decay_lr, use_amp=False,      #   │
                    log_interval=log_interval)                                  #   ╯ 
        training_time = time.time() - start_time
        print(f"\ntime: {training_time: .3f} s\n")
        

        print('plotting loss...')                                               # ◀─┬ Loss plotting
        models_eval.plotLoss(losses['train'], losses['val'],                    #   │
                            title='Loss during training', xlabel='steps',       #   │
                            ylabel='loss', show=False,                          #   │
                            save_path=(plot_path + "train_loss.png"))           #   │
        models_eval.plotLoss(losses['train_accuracy'], losses['val_accuracy'],  #   │
                            title='Accuracy during training',xlabel='steps',    #   │
                            ylabel='accuracy', show=False,                      #   │
                            save_path=(plot_path + "train_accuracy.png") )      #   ╯
        cnn.save()
    else:
        cnn.load()
        training_time = None

    
    print('computing accuracy...')
    y_true, y_pred = models_eval.getPredictions(cnn, test_loader, device)       # ◀─┬ Predictions over the test set
    accuracy = models_eval.computeAccuracy(y_true, y_pred )                     # ◀─┬ computing the accuracy
    print(f"accuracy: {accuracy: .4f}")                                         #   ╯

    print('compute confusion matrix...')
    cm_name = 'confusion_matrix.png'
    cm = models_eval.computeConfusionMatrix(y_true, y_pred, 
                                            num_classes=len(classes))
    models_eval.plotConfusionMatrix(cm, classes=classes, show=False,
                                    save_path=(plot_path + cm_name))
    


    print('saving experiment log...')
    log_file = save_path + cnn.name  + ".txt"

    save_experiment_log(
        filepath=log_file,
        model_name=name,
        config=config,
        n_params=n_parameters,
        accuracy=accuracy,
        training_time=training_time
    )