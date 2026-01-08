import torch
import matplotlib.pyplot as plt
import argparse
import datetime
import pydoc
from dataset.transformations import ChannelMean, Times255
import omegaconf
from omegaconf import OmegaConf


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



def parseArgumets():
    parser = argparse.ArgumentParser(description="Script for training and plotting")
    
    parser.add_argument('-train', type=str, default='True', 
                    help='Set to False to skip training')
    
    parser.add_argument('-config_path', type=str, default='Adam', 
                        help='path to the config file')

    args = parser.parse_args()

    config = {}
    config['do training'] = args.train.lower() in ('true', '1', 't', 'yes')
    config['config_path'] = args.config_path


    return config



def numberOfparameters(model):
    n = sum([p.numel() for p in model.parameters()])
    return n



def saveLog(filepath, model_name, config, 
                        n_params, accuracy, training_time):
    """ Appends a summary of the experiment to a text file. """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 50
    
    with open(filepath, "w") as f:
        f.write(f"\n{separator}\n")
        f.write(f"LOG: {timestamp}\n")
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
    """ Extracts normalization steps and a list of augmentation pipelines. """
    
    pipe_base    = OmegaConf.select(config, "data.transforms_base")            # 1. Parse Base steps
    pipe_base    = parseBlock(pipe_base) 
    
    pipes_aug = OmegaConf.select(config, "data.transforms")                    # 2. Parse Augmentations (list of pipelines)
    if pipes_aug:                                         
        pipes_aug = [parseBlock(p) for p in config.data.transforms]
    
    pipe_norm    = OmegaConf.select(config, "data.normalize")                   # 3. Parse Normalization (applied to everyone)
    pipe_norm    = parseBlock(pipe_norm)                                        
    
    return pipe_base, pipes_aug, pipe_norm



def parseBlock(block_config):
    """ Parses a single block containing a list of types and a list of params.
    Returns: A list of instantiated Python objects.  """
    
    transforms = []
    if block_config:
        types, params = block_config.types, block_config.params  

        for cls_str, args in zip(types, params):
            cls = pydoc.locate(cls_str)        
            kwargs = args if args else {}
            transforms.append(cls(**kwargs))
        
    return transforms

