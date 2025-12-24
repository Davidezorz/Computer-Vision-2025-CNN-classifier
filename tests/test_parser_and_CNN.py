from models.parser import convParser
from models.generalCNN import CNN
from dataset import dataloader
import torch

if __name__ == '__main__':

    settings =  """ 
                skip_store 
                conv2d     channels:   ->8   kernel_size: (3, 3)  padding: (1, 1)
                relu
                conv2d     channels:   ->8   kernel_size: (3, 3)  padding: (1, 1)
                relu      
                skip_add 
                maxpool2d                    kernel_size: (2, 2)  stride:  (2, 2)
                conv2d     channels:   ->16  kernel_size: (3, 3)  padding: (1, 1)
                skip_store
                relu
                conv2d     channels:   ->16  kernel_size: (3, 3)  padding: (1, 1)
                relu
                skip_add
                conv2d     channels:   ->16  kernel_size: (3, 3)  padding: (1, 1)  stride: (2, 2)
                relu
                maxpool2d                    kernel_size: (2, 2)  stride:  (2, 2)
                flatten
                linear     dims: ->50
                relu
                linear     dims: ->10 
                """
    parser = convParser()
    settings_dict = parser.str2dict(settings) 


    print("\nparser:\n")
    for s in settings_dict:
        print(s['type'], "\t", s['category'], "\n", s['args'], end="\n\n")
    
    X_train = torch.rand(28, 28)
    cnn = CNN(X_train.shape, settings_dict)
    print('cnn:\n\n', cnn)
    
    print('\nforward: ')
    logits = cnn.forward(X_train)
    print(logits)

    print('\nstore...')
    cnn.save()

    print('\nload...')
    cnn.load()
    """"""