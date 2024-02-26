import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lte.dmp.dmp_lte import DistributedModelParallelLTE
import lte.misc.distributed as D


class MultiheadLoRALinear(
        DistributedModelParallelLTE,
        nn.Linear,
        ):
    """
    Multihead Linear layer with LoRA [distributed-model parallel version]
    
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
        DistributedModelParallelLTE.__init__(
            self, 
            num_heads=num_heads, 
            lora_bias=lora_bias,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
        )

        for i in range(num_heads):
            device_id = f"cuda:{i % D.num_visible_devices()}"
            self.layers.append(nn.Linear(in_features, out_features, bias=bias, device=device_id))
            self.lora_A.append(nn.Linear(in_features, lora_r, bias=lora_bias, device=device_id))
            self.lora_B.append(nn.Linear(lora_r, out_features, bias=lora_bias, device=device_id))
            
        self.convert_to_lte_module()
        return

    def forward(self, inputs):   
        """
        Args:
            x (torch.Tensor): the input tensor
        Returns:
            outputs (torch.Tensor): the output tensor
        
        if not self.training then the forward pass is the same as the original Linear layer
        and uses the latest merged weights and biases.
        """
        if not self.training:
            outputs = F.linear(inputs.to(device=self.weight.device), self.weight, self.bias)
        else:
            if inputs.size(0) % self.num_heads != 0:
                raise ValueError("During training input size must be divisible by num_heads")
            xs = inputs.chunk(self.num_heads)
            outputs = []

            for x, layer, lora_A, lora_B in zip(xs, self.layers, self.lora_A, self.lora_B):
                x = x.to(device=lora_A.weight.device)
                s = self.scaling
                outputs.append(s * lora_B(lora_A(x)) + layer(x))

            outputs = torch.cat([x.to(device=inputs.device) for x in outputs])
        return outputs   

    @torch.no_grad()
    def compute_delta(self):
        """ computes the delta weight and bias for lora """
        (A, B), (b_A, b_B) = self.get_lora_params()

        delta_weight = self.scaling * B @ A
        delta_bias = None
        
        if self.lora_bias:
            delta_bias = self.scaling * (B @ b_A.unsqueeze(2) + b_B.unsqueeze(2)).squeeze(2)
        return delta_weight, delta_bias

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
