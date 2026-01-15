from parsing.inputParser import parseArgumets

from dataset.dataloader import DatasetManager
from utils.utils import getDevice, setupMatplotlib
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import utils
from utils import models_eval 
from torchvision import transforms
from omegaconf import OmegaConf
import pydoc
import torchvision
from torchvision.models import AlexNet_Weights
from sklearn.svm import SVC

from models.SVMs import MulticlassDAG_SVM, MulticlassECOC_SVM



class AlexNetFeatureExtractor(torchvision.models.AlexNet):
    def __init__(self):
        super().__init__()
        weights = AlexNet_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True))
        self.classifier[6] = nn.Identity()

        for param in self.parameters():                                            #   ╭  Freeze all 
            param.requires_grad = False                                            # ◀─┴  weights



def extractFeatures(loader, model, device):
    features_list = []
    labels_list = []
    
    model.eval()
    print(f"Extracting features from {len(loader)} batches...")
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        features = model(images)
        
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())
        
        if i % 10 == 0: print(f".", end="", flush=True)

    print()
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y








if __name__ == '__main__': 
    flags = parseArgumets()
    setupMatplotlib()
    models_dict = {'OVO': SVC,
                'DAG': MulticlassDAG_SVM,
                'ECOC': MulticlassECOC_SVM}
    config = OmegaConf.load(flags['config_path'])
    save_path = config.get('save_path', 'results/')                             # folder for saving plots

    np.random.seed(18)                                                          # ◀─╮
    torch.manual_seed(0)                                                        # ◀─┴ Setting the seed 

    image_dims = (224, 224)                                                     # ◀── Setting image resolution

    print('getting device...')                                                  #   ╮ Getting the 
    device = getDevice()                                                        # ◀─┴ deivce fan_in
    print(f'Using device: {device}')


    print('Loading AlexNet feature extractor...')
    cnn = AlexNetFeatureExtractor().to(device)
    n_parameters = utils.utils.numberOfparameters(cnn)


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


    print("\nFeature Extraction ")
    cnn.eval()
    with torch.no_grad():
        X_train, y_train = extractFeatures(train_loader, cnn, device)
        X_test, y_test = extractFeatures(test_loader, cnn, device)

    print(f"Training Features shape: {X_train.shape}")
    print(f"Testing Features shape:  {X_test.shape}")


    print("\nSVM Training ")
    kernel = config.model.get('kernel', 'linear')

    
    mode = config.model.get('mode', 'OVO')
    kwargs = config.model.get('additional_params', {})
    svm_class = models_dict[mode]
    print(f"using: {svm_class}\n")
    svm = svm_class(kernel=kernel, C=1.0, **kwargs)



    start_time = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time

    if svm_class == MulticlassECOC_SVM:
        print(svm.code_book)
        print()
        model_name = config.model.name
        plot_path = save_path + model_name
        svm.plotCodebook(classes=classes, show=False,
                         save_path=(plot_path + '_codebook'))
    print(f"SVM Training time: {training_time: .3f} s")


    print("\nEvaluation ")
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"accuracy: {accuracy: .4f}")


    print('\ncompute confusion matrix...')
    model_name = config.model.name
    plot_path = save_path + model_name
    cm_name = '_confusion_matrix.png'
    cm = models_eval.computeConfusionMatrix(y_test, y_pred, 
                                            num_classes=len(classes))
    models_eval.plotConfusionMatrix(cm, classes=classes, show=False,
                                    save_path=(plot_path + cm_name))
    
    print('saving log...')
    log_file = save_path + model_name + ".txt"

    utils.utils.saveLog(
        filepath=log_file,
        model_name=model_name,
        transforms_list=transforms_list,
        config=config,
        n_params= n_parameters,
        accuracy=accuracy,
        training_time=training_time
    )