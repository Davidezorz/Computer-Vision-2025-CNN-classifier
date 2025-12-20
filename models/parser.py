import re

import torch
import torch.nn as nn



# ╭───────────────────────────────────────────────────────────────────────────╮
# │                          Skip connection class                            │
# ╰───────────────────────────────────────────────────────────────────────────╯

class SkipHandle:
    """Simple container to share tensors between layers."""
    def __init__(self):
        self.val = None



class SkipStore(nn.Module):
    def __init__(self, handle):
        super().__init__()
        self.handle = handle

    def forward(self, x):
        self.handle.val = x                                                     # Store in shared handle
        return x



class SkipAdd(nn.Module):
    def __init__(self, handle):
        super().__init__()
        self.handle = handle

    def forward(self, x):
        assert self.handle.val is not None, "Skip val not stored!"
        return x + self.handle.val                                              # Retrieve and Add





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             convParser class                              │
# ╰───────────────────────────────────────────────────────────────────────────╯

class convParser:
    """
    Parses a string configuration into a dictionary definition for a 
    Convolutional Neural Network.

    Incoming string has structure:
        [module name] [arg]:[values] [arg]:[values] ...       <- important "\n"  
        [module name] [arg]:[values] [arg]:[values] ...
        ...

    [module name] should be a key of the self.[type]_modules
    [arg]         should be the name of the argument of that module
    [values]      could be:
                      - a single value
                      - a tuple
                      - [in_dim]->[out_dim] for convolution and linear layes
    
    Example:
        conv2d channels: 1->64 kernel_size: (3,3)
        skip_store
        linear dims:      ->10 bias: 0
        ...
        skip_add
    """

    def __init__(self):
        self.conv_modules     = {'conv2d': nn.Conv2d, 'maxpool2d': nn.MaxPool2d}
        self.linear_modules   = {'linear': nn.Linear}
        self.function_modules = {'relu':   nn.ReLU,   'softmax': nn.Softmax,
                                 'flatten': nn.Flatten}
        self.skip_modlules    = {'skip_store': SkipStore, 'skip_add': SkipAdd}

        self.map_modules = {'conv':     self.conv_modules,
                            'linear':   self.linear_modules, 
                            'function': self.function_modules,
                            'skip':     self.skip_modlules}
        
        self.map_arrows  = {'conv':     ('in_channels', 'out_channels'),
                            'linear':   ('in_features', 'out_features'), 
                            'function': (None, None),
                            'skip':     (None, None)}

        default = {'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1)}
        self.default  = {'conv':     default,                                   # set default values, needed for
                         'linear':   {},                                        # tensor shape inference
                         'function': {},
                         'skip':     {}                                                  
                        }
        
        self.skip = None


    def str2dict(self, configs_str: str):
        """function that performs the parsing"""
        configs = []
        for line in configs_str.split("\n"):                                    # iterate over lines: one line = one module
            line = line.strip()
            if not line: continue                                               # skip empty lines
                
            parts = [p.strip() for p in re.split(r"(\w+):", line)]
            module_type, category = parts[0].lower(), None

            for key, modules in self.map_modules.items():                       #  1. Identify Category and Class
                if module_type in modules:                                      #   ╭ Find the right modules container
                    category = key                                              # ◀─┤ Assign category (conv, linear...)
                    config   = self.default[key].copy()                         # ◀─┤ Load default args
                    module   = modules[module_type]                             # ◀─┴ Load the class (NOT INSTANCE)

            if category == 'skip':                                              # 2. Handle Skip Logic (Store vs Add)
                is_add = (module == SkipAdd)                                    #   ╭ Check if we are closing a skip
                assert self.skip or not is_add, "found skip_add without store"  # ◀─┴ Error: 'Add' before 'Store'
                self.skip = self.skip if is_add else SkipHandle()               #   ╭ Create Handle if Store, else reuse
                config['handle'] = self.skip                                    # ◀─┤ Share Handle between Store & Add
                self.skip =  None if is_add else self.skip                      #   ╰ Reset handle if we just closed it
            elif category is None:
                raise ValueError(f"Module '{module_type}' not supported")
            
            for type_str, value_str in zip(parts[1::2], parts[2::2]):           # 3. Parse Standard Arguments
                if '->' in value_str:                                           #   ╭ Check for arrow syntax
                    keys, values = self._parseArrow(value_str, category)        # ◀─┤ Parse arrow (e.g. 1->64)
                    config[keys[0]], config[keys[1]] = values[0], values[1]     # ◀─┴ Set in_dim and out_dim
                else:
                    value_str = value_str.replace("(", "").replace(")", "")     #   ╭ Clean tuples/ints
                    values = [int(v.strip()) for v in value_str.split(",")]     # ◀─┤ Convert string to list of ints
                    config[type_str] = values if len(values)>1 else values[0]   # ◀─┴ Store as tuple if len > 1
            
            configs.append({ 'type':        module_type, 
                             'category':    category,
                             'class':       module,
                             'args':        config })

        self._checkCorrectness(configs)
        return configs


    def _parseArrow(self, line: str, category: str):
        split  = [s.strip() for s in line.split("->")]
        values = [int(s) if s.isdigit() else -1 for s in split]                 # -1 means that should be inferred
        if values[1]==-1: raise ValueError("Output dimension must be defined ") # output cannot be inferred
        keys = self.map_arrows[category]
        assert keys[0], f"arrow not supported for category: {category}" 
        return keys, values
    

    def _checkCorrectness(self, configs):
        was_compatible = 1
        for config in configs:
            if (not was_compatible) and config['category'] == 'conv':
                raise Exception("convolutions are not " \
                                "supported after linear layers")
            if config['category'] == 'linear':
                was_compatible = 0
            

    def dict2str(self, configs_dict: dict):
        raise NotImplementedError