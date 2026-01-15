import torch
import matplotlib.pyplot as plt
import datetime
import pydoc
from dataset.transformations import ChannelMean, Times255

def getDevice(device: str = None) -> str:                                       #   ╭ Device auto
    """Selects the best available device or verifies the requested one."""      # ◀─┤ detection  
    if (device in [None, 'cuda']) and torch.cuda.is_available():                #   │
        return 'cuda'                                                           #   │
    if (device in [None, 'mps']) and torch.backends.mps.is_available():         #   │
        return 'mps'                                                            #   ╰
    return 'cpu'
    


def setupMatplotlib():
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.color'] = '#F9F9F9'




def numberOfparameters(model):
    n = sum([p.numel() for p in model.parameters()])
    return n




def saveLog(filepath, model_name, config, transforms_list,
                        n_params, accuracy, training_time):
    """ Appends a summary of the experiment to a text file. """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 50
    
    with open(filepath, "w") as f:
        f.write(f"\n{separator}\n")
        f.write(f"LOG: {timestamp}\n")
        f.write(f"{separator}\n")
        
        lr = config.training.get('lr')
        B = config.training.get('B')
        optim = config.training.get('optim')
        kernel = config.model.get('kernel')

        f.write(f"Model Name:      {model_name}\n")
        f.write(f"Parameters:      {n_params:,}\n") 
        if lr:     f.write(f"Learning Rate:   {lr}\n")
        if B:      f.write(f"Batch Size:      {B}\n") 
        if kernel: f.write(f"kernel:          {kernel}\n") 
        if optim:  f.write(f"Optimizer:       {optim}\n")
        f.write(f"Transforms:      \n") 
        for transforms in transforms_list:
            for transform in transforms:
                f.write(f"    - {transform}      \n") 
        f.write(f"Test Accuracy:   {accuracy:.4f}\n")

        training_time = f"{training_time: .4f}" if training_time else '-'
        f.write(f"training time:   {training_time}\n")
        f.write(f"{separator}\n\n")

    print(f"Log saved to {filepath}")



def processTransforms(config):
    """ Extracts normalization steps and a list of augmentation pipelines. """

    tranforms = config.data.transformations
    pipelines = []
    for string in ['base', 'train', 'test', 'normalization']:
        pipeline_type = tranforms.get(string)
        pipeline_type = pipeline_type if pipeline_type else []
        pipeline = [parseBlock(pipe) for pipe in pipeline_type]
        pipelines.append(pipeline)

    return pipelines


def parseBlock(pipe):   
    cls = pydoc.locate(pipe.types)  

    kwargs_str = pipe.get('params') if pipe.get('params') else {}
    
    kwargs = {}
    for key, value in kwargs_str.items():
        if isinstance(value, str) and '.' in value:
            obj = pydoc.locate(value)
            kwargs[key] = obj if obj is not None else value
        else:
            kwargs[key] = value

    #print(f"pipe.types: {pipe.types}   kwargs: {kwargs}")
    instance = cls(**kwargs)
    return instance



