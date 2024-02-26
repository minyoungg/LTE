import math
import torch
import torch.nn as nn
from lte import LTELayer


class MultiheadLoRALinear(nn.Linear, LTELayer):
    """
    Args:
        in_features (int): the number of input features
        out_features (int): the number of output features
        bias (bool): whether to use bias
        num_heads (int): the number of heads
        lora_r (int): the rank of LoRA 
        lora_alpha (int): the alpha value for LoRA
        lora_bias (bool): whether to use bias for LoRA
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias=True,
            num_heads: int = 2,
            lora_r: int = 1,
            lora_alpha: int = 1,
            lora_bias: bool = False,
            ):

        nn.Linear.__init__(self, in_features, out_features, bias)
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_bias = lora_bias
        self.scaling = self.lora_alpha / self.lora_r

        self.lora_A, self.lora_B = [], []

        for _ in range(num_heads):
            self.lora_A.append(nn.Linear(in_features, lora_r, bias=lora_bias))
            self.lora_B.append(nn.Linear(lora_r, out_features, bias=lora_bias))

        self.lora_A = nn.ModuleList(self.lora_A)
        self.lora_B = nn.ModuleList(self.lora_B)
        
        # store representation
        self._repr_A = list(self.lora_A)[0].__repr__()
        self._repr_B = list(self.lora_B)[0].__repr__()
        self.reset_lora_parameters()
        
        # disable training of original parameters
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        return    

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): the input tensor
        Returns:
            outputs (torch.Tensor): the output tensor
        """
        outputs = super().forward(x)
        for A, B in zip(self.lora_A, self.lora_B):
            outputs += self.scaling * B(A(x))            
        return outputs

    @torch.no_grad()
    def reset_lora_parameters(self):
        """ resets lora parameters. default is orthogonal initialization """

        def init_param(p):
            nn.init.orthogonal_(p)
            p.data *= math.sqrt(p.shape[1] / p.shape[0])
            return

        for lora_A, lora_B in zip(self.lora_A, self.lora_B):
            init_param(lora_A.weight.data)
            if self.lora_bias:
                nn.init.zeros_(lora_A.bias.data)

            nn.init.zeros_(lora_B.weight.data)
            if self.lora_bias:
                nn.init.zeros_(lora_B.bias.data)
        return
