import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import warnings

if torch.cuda.is_available():
    import flash_attn
    import flash_attn.layers.rotary


"""
╭ CONVENTIONS ─────────────────────────────────────────────────────────────────╮
│ ├─• B         batch size                                                     │
│ ├─• T         number of tokens in a batch i.e. length of a sequence/sentence │
│ ├─• C         embedding dimension of each token                              │
│ │                                                                            │
│ ├─• H         number of heads                                                │
│ ├─• V         vocabulary size (Not used in ViT, replaced by num_classes)     │
│ │                                                                            │
│ ╰─• patch     size of the square image patch                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
"""




# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                               Rotary  PE                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯


class Rotary(torch.nn.Module):
    def __init__(self, c, base=10_000):
        super().__init__()
        dtype = torch.get_default_dtype()
        inv_freq = 1. / (base ** (torch.arange(0, c, 2, dtype=dtype) / c))
        self.register_buffer('inv_freq', inv_freq)
        
        self.T_cached = 0                                                       # we will store the cos and sin values for the max T yet

    
    def forward(self, x):
        T = x.shape[1]
        dtype = self.inv_freq.dtype

        if self.T_cached < T:
            t = torch.arange(T, dtype=dtype, device=x.device)               # T
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())       # T c//2, first row is t[0]*inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)                         # T c

            self.cos = repeat(emb.cos(), 'T c -> 1 T 3 1 c')                # 1 T 3 1 c
            self.sin = repeat(emb.sin(), 'T c -> 1 T 3 1 c')                # 1 T 3 1 c
            
            self.cos[:,:,2,:,:].fill_(1.)                                   # ◀─┬ This makes the transformation 
            self.sin[:,:,2,:,:].fill_(0.)                                   # ◀─╯ on values an identity
            
            self.T_cached = T                                               # update T_cached

        cos = self.cos[:, :T, :, :, :]                                      # ◀─┬ cut based on the 
        sin = self.sin[:, :T, :, :, :]                                      # ◀─╯ token length

        return cos, sin



def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                           Patch Projection                                   │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class PatchProjection(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, C: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=patch_size, stride=patch_size), # ◀─ Convolve to get patches
            Rearrange('B C H W -> B (H W) C'),                                    # ◀─ Flatten spatial dims to T
        )    

    def forward(self, x):                                                         # B C H W
        x = self.projection(x)    
        return x                                                                  # B T C





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                               Layer Norm                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯


class LayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([C]))
        self.bias  = nn.Parameter(torch.zeros([C]))
        self.C     = C
    
    
    def forward(self, x):
        x = F.layer_norm(x.float(), [self.C])
        return x * self.scale[None, None, :] + self.bias[None, None, :]





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                            Multi Head Attention                              │
# ╰──────────────────────────────────────────────────────────────────────────────╯


class MultiHeadAttention(nn.Module):
    def __init__(self, C: int = 256, H: int = 8, p_dropout: float = 0.1):
        super().__init__()
        assert C % H == 0, "embedding dimension C must be divisible by the number of heads H"
        
        self.C, self.H = C, H
        self.c         = int(C // H)

        self.W_qkv     = nn.Linear(C, 3 * C, bias=False)
        self.W_o       = nn.Linear(C, C) 
        self.dropout   = nn.Dropout(p_dropout)

        cuda = torch.cuda.is_available()
        self.attention = self._attention_cuda if cuda else self._attention
        

    def forward(self, x, rotary_cos_sin, seqlens):
        """ 
        x:              B, T C, tensor input
        rotary_cos_sin: cosine and sine tensor from Rotary class
        seqlens:        B T, how long is each sequence 
        """
        qkv = self.W_qkv(x)                                                     # B T (three C)
        qkv = rearrange(qkv, 'B T (three H c) -> B T three H c',                # B T three H c
                        three=3, H=self.H)
        
        x = self.attention(qkv, rotary_cos_sin, seqlens)                        # B T C 
        x = self.dropout(self.W_o(x))                                           # B T C 

        return x    


    def _attention(self, qkv, rotary_cos_sin, seqlens):
        cos, sin = rotary_cos_sin                                               #  ╭ rotary positional embedding 
        qkv = qkv * cos + rotate_half(qkv) * sin                                # ◀╯ B T three H c  

        qkv = rearrange(qkv, 'B T three H c -> B three H T c')
        q, k, v = qkv.unbind(dim=1)                                             # B three H S c -> 3 * B H T c

        c = q.shape[-1]                                                         #  ╭ compute attention
        attn_scores = (q @ k.transpose(-2, -1)) * (c ** -0.5)                   # ◀┤ B H T T
        attn_probs = F.softmax(attn_scores, dim=-1)                             # ◀┤ B H T T
        x = attn_probs @ v                                                      # ◀╯ B H T c

        return rearrange(x, 'B H T c -> B T (H c)')                             # B T C


    def _attention_cuda(self, qkv, rotary_cos_sin, seqlens):
        B, T, _, H, c = qkv.shape
        dv            = qkv.device

        cos, sin = rotary_cos_sin                                                     #  ╭ rotary positional embedding 
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2].to(qkv.dtype)                        # ◀┤  T c//2
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2].to(qkv.dtype)                        # ◀┤  T c//2
        qkv = flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)           # ◀╯  B T 3 H c
        
        qkv = rearrange(qkv, 'B T ... -> (B T) ...')                                  # (B T) 3 H c
        cu_seqlens = seqlens.cumsum(-1) if seqlens else self.cu_seqlens(B, T, dv)     # ◀─ B + 1, compute the cumulative sequence length

        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(         #  ╭ compute attention 
            qkv, cu_seqlens, T, 0., causal=False)                                     # ◀╯ (B T) 3 H c
        
        return rearrange(x, '(B T) H c -> B T (H c)', B=B)                            # B T C
        

    def cu_seqlens(self, B, T, device):
        return torch.arange(0, (B + 1)*T, T, dtype=torch.int32, device=device)
        




# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                            Feed Forward Network                              │
# ╰──────────────────────────────────────────────────────────────────────────────╯
# (Retained exactly from Code 1)

class FeedForward(nn.Module):
    def __init__(self, C: int = 64, factor: int = 4):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(C, factor*C),
            nn.GELU(approximate='tanh'),
            nn.Linear(factor*C, C)
        )
    
    def forward(self, x):
        return self.FFN(x)





# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                   Blocks                                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# (Modified from DiTBlock to ViTBlock: Removed conditioning/ALN)

class ViTBlock(nn.Module):                                                      # Paper: ┬ An Image is Worth 16x16 Words
    def __init__(self, C, H, p_dropout=0.1, FFN_ratio=4):                       #        ╰ https://arxiv.org/abs/2010.11929
        super().__init__()
        self.H         = H

        self.norm1     = LayerNorm(C)
        self.attention = MultiHeadAttention(C, H)

        self.norm2     = LayerNorm(C)
        self.FFN       = FeedForward(C, FFN_ratio)
        self.dropout   = nn.Dropout(p_dropout)


    def forward(self, x, rotary_cos_sin, seqlens=None):
        # Standard Pre-Norm ViT architecture
        
        x_norm = self.norm1(x)                                                  # ◀─ Normalize
        x = x + self.attention(x_norm, rotary_cos_sin, seqlens)                 # ◀─ Attention + Skip connection
        
        x_norm = self.norm2(x)                                                  # ◀─ Normalize
        x = x + self.dropout(self.FFN(x_norm))                                  # ◀─ FFN + Skip connection

        return x
    




# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                             Vision Transformer                               │
# ╰──────────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class VisionTransformer(nn.Module):                                           #huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, 
                patch_size: int = 4,     # ◀ height/width of patch
                num_classes: int = 10,   # ◀ number of output classes
                in_channels: int = 3,    # ◀ image channels (3 for RGB)
                C: int = 128,            # ◀ embedding dimension
                H: int = 4,              # ◀ number of heads
                N: int = 3,              # ◀ number of blocks
                p: float = 0.1,          # ◀ probability of dropout
                name = 'vit'
                ):
        super().__init__()
        self.name = name
        # Patch Embedding
        self.patch_projection = PatchProjection(in_channels, patch_size, C)
        
        # Class Token (Learnable)
        self.class_token = nn.Parameter(torch.randn(1, 1, C) * 0.02)
        
        # Rotary Positional Embedding (Replaces the learnable/sinusoidal PE)
        self.rotary    = Rotary(C // H)
        
        # Transformer Blocks
        blocks         = [ViTBlock(C, H, p) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        # Output Head
        self.norm_final = LayerNorm(C)
        self.head       = nn.Linear(C, num_classes)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()


    def save(self, folder='.weights/', name=None):                              # ◀┬─ save model 
        name = name if name else self.name                                      #  │  weights
        file = folder + name + '.pth'                                           #  │  
        torch.save(self.state_dict(), file)                                     #  ╯


    def load(self, folder='.weights/', name=None):                              # ◀┬─ load model
        name = name if name else self.name                                      #  │  weights
        file = folder + name + '.pth'                                           #  │  
        try:                                                                    #  │ 
            self.load_state_dict(torch.load(file, weights_only=True))           #  │
            print('model loaded')                                               #  │
        except Exception as e:                                                  #  │
            warnings.warn("Model weights not avaiable \n\n")                    #  ╯


    def forward(self, x):
        # x: B C H W

        x = self.patch_projection(x)                                            # B T_patches C

        B = x.shape[0]
        cls_token = repeat(self.class_token, '1 1 C -> B 1 C', B=B)
        x = torch.cat([cls_token, x], dim=1)                                    # B (T_patches + 1) C
        
        rotary_cos_sin = self.rotary(x)
        
        for block in self.blocks:                                               # 4. Transformer Blocks
            x = block(x, rotary_cos_sin, seqlens=None)

        # 5. Final Classification
        x = self.norm_final(x)
        x = x[:, 0, :]                                                          # Take CLS token only
        x = self.head(x)

        return x
      
    
    @torch.no_grad()
    def predict(self, x):                                                       # ◀┬─ performs inference by
        self.eval()                                                             #  │  taking the most probable
        logits = self(x)                                                        #  │  label (doesn't compute 
        return logits.argmax(dim=-1)                                            #  ╯  the softamx)
