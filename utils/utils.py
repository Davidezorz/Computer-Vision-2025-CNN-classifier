import torch
                                                         
def getDevice(device: str = None) -> str:                                       #   ╭ Device auto
    """Selects the best available device or verifies the requested one."""      # ◀─┤ detection  
    if (device in [None, 'cuda']) and torch.cuda.is_available():                #   │
        return 'cuda'                                                           #   │
    if (device in [None, 'mps']) and torch.backends.mps.is_available():         #   │
        return 'mps'                                                            #   ╰
    return 'cpu'
    