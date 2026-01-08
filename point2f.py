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







class CNN_ensemble(nn.Module):
    def __init__(self, models: list, name='CNN_ensemble'):
        super().__init__()
        self.models  = nn.ModuleList(models) 
        self.N       = len(models)
        self.out_dim = models[0].current_dims[0]
        self.in_dims = models[0].in_dims
        self.name = name
        print(self.out_dim)
    

    def probabilities(self, x):
        if x.ndim < 4:
            x = x.view(1, *self.in_dims)

        logits_list = torch.zeros(x.shape[0], self.out_dim, self.N, 
                                  device=x.device)
        
        for i, model in enumerate(self.models):
            logits_list[:, :, i] = model(x)

        probs_list = torch.softmax(logits_list, dim=1)
        avg_probs = probs_list.mean(dim=-1)
        
        return avg_probs
    

    @torch.no_grad()
    def predict(self, x):
        self.eval() 
        probs = self.probabilities(x)
        return probs.argmax(dim=-1)                                         #  ╯  the softamx)




if __name__ == '__main__': # 18
    N = 9

    flags = parseArgumets()
    config = OmegaConf.load(flags['config_path'])
    setupMatplotlib()

    np.random.seed(18)                                                          # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed 

    image_dims = [64, 64]                                                       # ◀── Setting image resolution
    
    print('parsing...')                                                         #   ╮ Converting the string into 
    parser = convParser()                                                       #   │ a dictionary config using 
    nn_config = parser.str2dict(config.model.nn_config)                         # ◀─┴ the convParser class


    print('cnn definition...')                                                  #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce fan_in
    print(f'Using device: {device}')


    init_type = pydoc.locate(config.model.init.type)                            # ◀─┬ define the initialization class
    init_conf = OmegaConf.to_container(config.model.init.conf, resolve=True)    # ◀─┴ and its config

    models = []
    for i in range(N):
        name = config.model.name + "_" + str(i)
        cnn  = CNN(image_dims, nn_config, name=name, init_type=init_type,           # ◀─┬ Instantiating the Convolutional 
                    init_conf=init_conf).to(device)                               #   ╯ Neural Network
        models.append(cnn)
    print(f"first cnn: {models[0]}")

    n_parameters = utils.utils.numberOfparameters(models[0])
    print(f"Parameters: {n_parameters}\n")

    
    print('getting data...')
    folder_path = '.data/'                                                      # ◀─┬ define the folder path
    B = config.training.B                                                       # ◀─┴ and the batch size
    
    pipe_base, pipes_aug, pipe_norm = utils.utils.processTransforms(config)
    dataset_mng = DatasetManager(folder_path, image_dims, val_split=0.15,       # ◀─┬ Instantiating the class
                                 pipe_base = pipe_base,                         #   │ that retrive dataloaders
                                 pipes_aug = pipes_aug,                         #   │ 
                                 pipe_norm = pipe_norm)                         #   ╯  
    
    data_loaders, classes = dataset_mng.get(B)                                  #   ╮ Getting the
    train_loader, val_loader, test_loader = data_loaders                        # ◀─┴ dataloaders

    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader:   {len(val_loader)}")
    print(f"test_loader:  {len(test_loader)}")

    save_path = 'results/point2e/'                                               # folder for saving plots
    start_time = time.time()
    training_time = None

    accuracies = []
    cms = []
        

    for i, cnn in enumerate(models):
        plot_path = save_path + cnn.name 
        if flags['do training']:                                                    # ◀── TRAINING LOOP 
            optim_class = pydoc.locate(config.training.optim)                       # ◀── define the optimizer class
            optim_opt = OmegaConf.to_container(config.training.opt, resolve=True)
            lr = config.training.lr
            decay_lr = config.training.decay_lr
            log_interval = config.training.log_interval
            losses = train(cnn, train_loader, val_loader, patience=20, lr=lr,       # ◀─┬ Training loop  
                        device=device, epochs=12*3,  optim_class=optim_class,       #   │
                        optim_opt=optim_opt, decay_lr=decay_lr, use_amp=False,      #   │
                        log_interval=log_interval)                                  #   ╯ 
  
            
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

    training_time = time.time() - start_time
    print(f"\ntime: {training_time: .3f} s\n")

    
    print('computing single model accuracy...')
    for i, cnn in enumerate(models):
        y_true, y_pred = models_eval.getPredictions(cnn, test_loader, device) # ◀─┬ Predictions over the test set
        accuracy = models_eval.computeAccuracy(y_true, y_pred )                     # ◀─┬ computing the accuracy
        print(f"accuracy {i}: {accuracy: .4f}")
    print("\n")

    cnns = CNN_ensemble(models)
    plot_path = save_path + cnns.name

    print('computing accuracy...')
    y_true, y_pred = models_eval.getPredictions(cnns, test_loader, device)      # ◀─┬ Predictions over the test set
    accuracy = models_eval.computeAccuracy(y_true, y_pred )                     # ◀─┬ computing the accuracy
    print(f"accuracy: {accuracy: .4f}")                                         #   ╯

    print('compute confusion matrix...')
    cm_name = 'confusion_matrix.png'
    cm = models_eval.computeConfusionMatrix(y_true, y_pred, 
                                            num_classes=len(classes))

    models_eval.plotConfusionMatrix(cm, classes=classes, show=False,
                                    save_path=(plot_path + cm_name))



    print('saving experiment log...')
    log_file = save_path + cnns.name  + ".txt"

    utils.utils.saveLog(
        filepath=log_file,
        model_name=name,
        config=config,
        n_params=n_parameters,
        accuracy=accuracy,
        training_time=training_time
    )