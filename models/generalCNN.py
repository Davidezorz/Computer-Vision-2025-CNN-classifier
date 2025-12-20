import torch
import torch.nn as nn
import math
from . import parser

class CNN(nn.Module):

    def __init__(self, image_dims, configs):
        super().__init__()
        blocks = []
        image_dims = list(image_dims)

        self.current_dims = image_dims if len(image_dims)>2 else [1] + image_dims
        self.in_dims = self.current_dims.copy()
        print(self.current_dims)

        for i, config in enumerate(configs):
            print(config['type'])

            self._inferredOutDim(config['args'], config['category'])
            module = config['class'](**config['args'])
            blocks.append(module)

            print(self.current_dims)

        self.blocks    = nn.ModuleList(blocks)
        self.apply(self._initWeights)


    def _inferredOutDim(self, kwargs, category):
        if category == 'conv':
            C, H, W = self.current_dims
            if kwargs.get('in_channels') == -1:
                kwargs['in_channels'] = self.current_dims[0]
            self.current_dims[0] = kwargs.get('out_channels', C)
            self.current_dims[1] = self._outDim([H, W], kwargs, 0)
            self.current_dims[2] = self._outDim([H, W], kwargs, 1)
        elif category == 'linear':
            if kwargs.get('in_features') == -1:
                kwargs['in_features'] = math.prod(self.current_dims)
            self.current_dims = [kwargs['out_features']]



    def _outDim(self, in_dim, kwargs, idx):
        coef1 = in_dim[idx] + 2*kwargs['padding'][idx]
        coef2 = kwargs['dilation'][idx] * (kwargs['kernel_size'][idx]-1)
        coef3 = (coef1 - coef2 - 1)/kwargs['stride'][idx]

        return math.floor(coef3 + 1)


    def _initWeights(self, module):
        """Applies Kaiming initialization to Linear and Conv layers."""
        if isinstance(module, nn.Linear):                                       # Initialize Linear Layers
            nn.init.kaiming_uniform_(module.weight, mode='fan_in')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Conv2d):                                     # Initialize Convolutional Layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def _applySkipConnection(self, x, skip):
        if skip:
            return skip(x)
        return x


    def forward(self, x):
        if x.ndim < 4:                                                          # shape the tensor correctly: 
            x = x.view(1, *self.in_dims)                                        # B start_channels H W
        
        skip = None

        for i, block in enumerate(self.blocks[:-1]):
            print(f"{i}: {x.shape}")
            x = self._applySkipConnection(x, skip)
            x = block(x)

            if isinstance(block, parser.Skip):
                skip = block

        x = self._applySkipConnection(x, skip)
        logits = self.blocks[-1](x)
        return logits
    
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1)