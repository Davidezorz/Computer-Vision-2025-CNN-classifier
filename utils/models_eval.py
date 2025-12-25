import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def getPredictions(model, loader, device):
    """
    Iterates over the dataloader to generate predictions for the entire dataset.
    
    Args:
        model: The trained PyTorch model (CNN).
        loader: The DataLoader containing the dataset.
        device: The device to run inference on (CPU or CUDA).
        
    Returns:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
    """

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            
            pred = model.predict(X)                                             # ◀── Get the predictions

            y_true.extend(y.cpu().numpy())                                      # ◀─╭ Move to CPU and convert to
            y_pred.extend(pred.cpu().numpy())                                   # ◀─┴ numpy, then extend lists
            
    return y_true, y_pred




def computeConfusionMatrix(y_true, y_pred, num_classes=15):
    """ Computes the confusion matrix using sklearn. """
    cm = confusion_matrix(y_true, y_pred)
    return cm



def computeAccuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return (y_true==y_pred).mean()



def plotConfusionMatrix(cm, classes=None, show=True, save_path=None):
    """
    Plots the confusion matrix using Seaborn.
    
    Args:
        cm: The confusion matrix array.
        classes: Optional list of class names (strings) for the axis labels.
    """
    plt.figure(figsize=(18, 14))
    classes = classes if classes else [str(i) for i in range(cm.shape[0])]      # use indices if classes is None   
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',                         # ◀─┬ Create a heatmap with color map 'cmap'
                xticklabels=classes,                                            #   │  - annot=True show the numbers inside the squares
                yticklabels=classes)                                            #   │  - fmt='d' ensures numbers are integers
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    _ = plt.show() if show else plt.close()





def plotLoss(y1, y2, show=True, save_path=None, title='loss plot',
             xlabel = 'steps' , ylabel='loss'):
    y1, y2 = np.array(y1), np.array(y2)
     
    x = np.arange(y1.shape[0])
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.plot(x, y1, label='train', color='blue')
    ax.plot(x, y2, label='validation', color='purple')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    y_max = max(y1.max(), y2.max())*1.1
    y_min = min(y1.min(), y2.min())*0.9
    ax.set_ylim(y_min, y_max)

    if save_path: fig.savefig(save_path, bbox_inches='tight')
    _ = plt.show() if show else plt.close(fig)
    return ax