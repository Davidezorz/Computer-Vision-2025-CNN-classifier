from parsing.inputParser import parseArgumets

from dataset.dataloader import DatasetManager
from train import train
from utils.utils import getDevice, setupMatplotlib
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
import torchvision
from torchvision import models
from torchvision.models import AlexNet_Weights



class AlexNet(torchvision.models.AlexNet):
    def __init__(self, num_classes=1000, weights=None):
        super().__init__(num_classes=1000)
        self.name = 'AlexNet'

        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=True))

        if num_classes != 1000:
            num_features = self.classifier[6].in_features                                  #   ╮ Replace the last fully
            self.classifier[6] = nn.Linear(num_features, num_classes)  
        
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self(x)
        return logits.argmax(dim=-1)                                                        #  ╯  the softamx)




def getAlexnet(num_classes):
    """ https://docs.pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html 
        https://pytorch.org/hub/pytorch_vision_alexnet/ """
    print("Loading AlexNet...")                                                 # Load AlexNet with pre-trained ImageNet weights
    weights = AlexNet_Weights.DEFAULT
    model   = AlexNet(num_classes=num_classes, weights=weights)

    for param in model.parameters():                                            #   ╭ Freeze all weights
        param.requires_grad = False                                             # ◀─┴ connected layer
    
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model





if __name__ == '__main__': # 18
    flags = parseArgumets()
    setupMatplotlib()
    config = OmegaConf.load(flags['config_path'])

    np.random.seed(18)                                                          # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed 

    image_dims = (224, 224)                                                     # ◀── Setting image resolution


    print('cnn definition...')                                                  #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce fan_in
    print(f'Using device: {device}')


    cnn  = getAlexnet(15).to(device)
    cnn_name = config.model.name
    print(cnn)

    for name, param in cnn.named_parameters():
        print(name, "\t", param.requires_grad)

    n_parameters = utils.utils.numberOfparameters(cnn)
    print(f"Parameters: {n_parameters}\n")

    
    print('getting data...')
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
    print(f"test_loader:  {len(test_loader)}")

    save_path = config.get('save_path', 'results/')                            # folder for saving plots
    plot_path = save_path + cnn_name 

    if flags['do training']:                                                    # ◀── TRAINING LOOP 
        params_to_update = []
        for name, param in cnn.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print(f"\tTraining will update: {name}")

        start_time = time.time()
        optim_class = pydoc.locate(config.training.optim)                       # ◀── define the optimizer class
        optim_opt = OmegaConf.to_container(config.training.opt, resolve=True)
        lr = config.training.lr
        decay_lr = config.training.decay_lr
        log_interval = config.training.log_interval
        losses = train(cnn, train_loader, val_loader, patience=30, lr=lr,       # ◀─┬ Training loop  
                    device=device, epochs=12*3,  optim_class=optim_class,       #   │
                    optim_opt=optim_opt, decay_lr=decay_lr, use_amp=False,      #   │
                    log_interval=log_interval, parameters=params_to_update)     #   ╯ 
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
    else:
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
    

    models_eval.storeExamples(cnn, train_loader, test_loader, 
                              classes, plot_path, device)

    print('saving experiment log...')
    log_file = save_path + cnn_name + ".txt"

    utils.utils.saveLog(
        filepath=log_file,
        model_name=cnn.name,
        transforms_list=transforms_list,
        config=config,
        n_params=n_parameters,
        accuracy=accuracy,
        training_time=training_time
    )