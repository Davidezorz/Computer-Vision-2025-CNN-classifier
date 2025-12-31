import torch
import matplotlib.pyplot as plt
import argparse

                                                         
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
    args = parser.parse_args()

    flags = {}
    flags['do training'] = args.train.lower() in ('true', '1', 't', 'yes')

    return flags
    
