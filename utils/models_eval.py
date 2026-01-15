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
             xlabel = 'steps' , ylabel='loss', line_at=None, 
             ylim=None, r: float = 1):
    y1, y2 = np.array(y1), np.array(y2)
    y_tot = np.hstack([y1, y2])
    ylim = ylim if ylim != None else [np.min(y_tot), 1.1*np.max(y_tot)]
     
    x = np.arange(y1.shape[0])*r

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.plot(x, y1, label='train', color='#561A66')
    ax.plot(x, y2, label='validation', color='#FF8D13', linestyle='dashed')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(*ylim)
    if line_at is not None: ax.vlines(line_at*r, *ylim, linestyle='dashed', 
                                      color="#0F0F0F",
                                      label='best validation')
    
    ax.legend()
    if save_path: fig.savefig(save_path, bbox_inches='tight')
    _ = plt.show() if show else plt.close(fig)
    return ax






def plotExamples(X, y, y_preds=None, save_path="", class_names=None):
    """
    Saves a 3x3 grid of images comparing True vs Predicted labels.
    
    Args:
        X:           Tensor  Batch of images.
        y:           Tensor  True labels.
        y_pred:      Tensor  Predicted labels (optional).
        save_path:   str     The filename/path to save the resulting image.
        class_names: list    List of class names mapping to label indices.
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]

        if i >= len(X): 
            ax.axis('off')
            continue

        img = X[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img, cmap='gray')
        
        y_true = y[i].item()
        true_text = class_names[y_true] if class_names else str(y_true)

        title_color = 'black'
        title = f"True: {true_text}"

        if y_preds is not None:
            y_pred = y_preds[i].item()
            pred_text = class_names[y_pred] if class_names else str(y_pred)
            title = title +  f"\nPred: {pred_text}"
            title_color =  'green' if y_true == y_pred else 'red'

        ax.set_title(title, fontsize=12, color=title_color, fontweight='bold')
        ax.axis('off') 

    plt.tight_layout()
    file_name = save_path + '_examples.png'
    plt.savefig(file_name)
    print(f"Grid saved successfully to: {file_name}")
    plt.close(fig)




@torch.no_grad()
def storeExamples(cnn, train_loader, test_loader, 
                  classes, plot_path, device):
    
    temp_loader = torch.utils.data.DataLoader(
        test_loader.dataset, 
        batch_size=9, 
        shuffle=True 
    )

    loaders = [train_loader, temp_loader]
    for loader, type_loader in zip(loaders, ['_train', '_test']):
        X, y = next(iter(loader))
        X, y = X.to(device), y.to(device)

        y_pred=cnn.predict(X)
        plotExamples(X, y, y_pred,
                    save_path=plot_path + type_loader, 
                    class_names=classes)