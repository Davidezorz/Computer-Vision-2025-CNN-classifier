import re

import torch
import torch.nn as nn

"""
class NN:

    def __init__(self):
        pass

    def Conv2d(self):
        pass
    def MaxPool2d(self):
        pass
    def Linear(self):
        pass
    def ReLU(self):
        pass
    def Softmax(self):
        pass
    def Flatten(self):
        pass

nn = NN()
""" 

class Skip(nn.Module):

    def __init__(self, steps):
        super().__init__()
        self.x     = None
        self.steps = steps


    def _store(self, x):
        self.x = x
        print('stored skip :)')


    def _forward(self, x):
        print('applied skip :)')
        assert self.x is not None, "value not stored"
        return self.x + x


    def forward(self, x):
        self.steps -= 1
        if self.steps == 0:
            return self._forward(x)
        if self.x is None:
            self._store(x)
        return x





class convParser:
    """
    Class for parsing the arguments of a string that contains the covolutional 
    NN structure.

    Incoming string has structure:
        [module name] [arg]:[values] [arg]:[values] ...       <- important "\n"  
        [module name] [arg]:[values] [arg]:[values] ...
        ...

    Then one line will be one actuall module, with exception of the "skip_add",
    that will change the initialization settings of the Skip class

    [module name] should be a key of the self.[type]_modules
    [arg]         should be the name of the argument of that module
    [values]      could be:
                      - a single value
                      - a tuple
                      - [in_dim]->[out_dim] for convolution and linear layes
    """

    def __init__(self):
        self.conv_modules     = {'conv2d': nn.Conv2d, 'maxpool2d': nn.MaxPool2d}
        self.linear_modules   = {'linear': nn.Linear}
        self.function_modules = {'relu':   nn.ReLU,   'softmax': nn.Softmax,
                                 'flatten': nn.Flatten}
        self.skip_keys        = ['skip_store', 'skip_add']
        self.skip_modlules    = {self.skip_keys[0]: Skip, 
                                 self.skip_keys[1]: None}

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
                         'linear':   {},                                        # resolution computation
                         'function': {},
                         'skip':     {'steps': -1}                                                  
                        }


    def str2dict(self, configs_str: str):
        """function that performs the parsing"""
        configs = []
        skip    = None
        for i, line in enumerate(configs_str.split("\n")):                      # iterate over lines: one line, one module
            line = line.strip()
            if not line: continue                                               # skip empty lines
                
            parts = [p.strip() for p in re.split(r"(\w+):", line)]
            module_type, category = parts[0].lower(), None

            if module_type == self.skip_keys[-1]:                               # ◀─┬ Exception of the skip_add instruction: 
                assert skip, "skip need to stored a value before"               #   ╭ The "step" that the Skip instance
                skip['config']['steps'] = i - skip['i'] + 1                     # ◀─┤ should wait is the difference of current 
                skip = None                                                     #   │ iteration minus the iteration at which  
                continue                                                        #   ╰ it was stored

            for key, modules in self.map_modules.items():
                if module_type in modules:
                    category = key
                    config   = self.default[key].copy()
                    module   = modules[module_type]

                    if module_type == self.skip_keys[0]:
                        assert skip is None, "multiple skip not supported"
                        skip = {'config': config, 'i': i}

            if category is None:
                raise ValueError(f"Type of module '{module_type}' not supported")
            
            for type_str, value_str in zip(parts[1::2], parts[2::2]):
                if '->' in value_str:
                    keys, values = self._parseArrow(value_str, category)
                    config[keys[0]], config[keys[1]] = values[0], values[1]
                else:
                    value_str = value_str.replace("(", "").replace(")", "")
                    values = [int(v.strip()) for v in value_str.split(",")]
                    config[type_str] = values if len(values)>1 else values[0]
            
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