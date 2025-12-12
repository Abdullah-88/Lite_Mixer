import torch
from torch import nn
from einops.layers.torch import Rearrange


class VectorDynamicTanh(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.alpha = nn.Parameter(torch.randn(input_shape))
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x

     
class GatingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.proj_1 =  nn.Linear(dim,dim,bias=False)
        self.proj_2 =  nn.Linear(dim,dim,bias=False)
            
        self.silu = nn.SiLU()
       
             	   
    def forward(self, x):

        u, v = x, x 
        u = self.proj_1(u)
        u = self.silu(u)
        v = self.proj_2(v)
        g = u * v
        
        return g

class LiteMixerBlock(nn.Module):

    def __init__(self, dim, num_patch):
        super().__init__()
        
        self.norm =  VectorDynamicTanh(dim)
       

        self.context_process = nn.Sequential(
           
            Rearrange('b n d -> b d n'),
            GatingUnit(num_patch),
            Rearrange('b d n -> b n d')
        )

        self.token_process = GatingUnit(dim)
        

    def forward(self, x):
        
        residual = x
    
        x = self.norm(x)
    
        x = self.context_process(x)
    
        x = x + residual
    
        residual = x
    
        x = self.norm(x)    
    
        x = self.token_process(x)
    
        x = x + residual

        return x


class LiteMixer(nn.Module):
    def __init__(self, d_model,num_patch, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[LiteMixerBlock(d_model,num_patch) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)







