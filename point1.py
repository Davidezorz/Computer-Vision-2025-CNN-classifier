from models.parser import convParser
from models.generalCNN import CNN
from dataset.dataloader import DatasetManager
from train import train
from utils.utils import getDevice
import torch
import time
from utils import models_eval 










if __name__ == '__main__':
    torch.manual_seed(0)
    image_dims = (64, 64)
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
    
    print('parsing...')
    parser = convParser()
    config = parser.str2dict(config_str)


    print('cnn definition...')
    device = getDevice()
    print(f'Using device: {device}')
    init_type =torch.nn.init.normal_
    init_conf = {'conv': {'std': 0.01}, 'linear': {'std': 0.01}}
    cnn    = CNN(image_dims, config, name='CNN_1', init_type=init_type, 
                  init_conf=init_conf).to(device)


    print('getting data...')
    folder_path = '.data/'
    B = 32
    dataset_mng = DatasetManager(folder_path, image_dims, val_split=0.15,
                                 normalize=False)

    data_loaders, classes = dataset_mng.get(B)
    train_loader, val_loader, test_loader = data_loaders

    for x, y in train_loader: # if debugging needed
        break
    
    """ """
    start = time.time()
    optim_class = torch.optim.SGD
    optim_opt   = {'momentum': 0.9}
    losses = train(cnn, train_loader, val_loader, patience=20, 
                   device=device, epochs=30, lr=5e-4, optim_class=optim_class,
                   optim_opt=optim_opt, use_amp=False)
    print(f"\ntime: {time.time()-start: .3f} s\n")
    

    print('plotting loss...')
    models_eval.plotLoss(losses['train'], losses['val'], 
                         title='Loss during training',
                         xlabel='steps', ylabel='loss')
    models_eval.plotLoss(losses['train_accuracy'], losses['val_accuracy'], 
                         title='Accuracy during training',
                         xlabel='steps', ylabel='accuracy')

    print('computing accuracy...')
    y_true, y_pred = models_eval.get_all_predictions(cnn, test_loader, device)
    accuracy = models_eval.computeAccuracy(y_true, y_pred )
    print(f"accuracy: {accuracy: .4f}")

    print('compute confusion matrix...')
    y_true, y_pred = models_eval.get_all_predictions(cnn, test_loader, device)
    cm = models_eval.compute_confusion_matrix(y_true, y_pred, num_classes=len(classes))
    models_eval.plot_confusion_matrix(cm, classes=classes)
    


