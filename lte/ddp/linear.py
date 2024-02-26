import math
import torch
import torch.nn as nn
from lte.ddp.ddp_lte import DistributedDataParallelLTE


class MultiheadLoRALinear(
        DistributedDataParallelLTE,
        nn.Linear,
        ):
    """
    Multihead Linear layer with LoRA [distributed-data parallel version]
    
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
        DistributedDataParallelLTE.__init__(
            self, 
            num_heads=num_heads, 
            lora_bias=lora_bias,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
        )

        for _ in range(num_heads):
            self.lora_A.append(nn.Linear(in_features, lora_r, bias=lora_bias))
            self.lora_B.append(nn.Linear(lora_r, out_features, bias=lora_bias))

        self.convert_to_lte_module()
        return

    def forward(self, x):   
        """
        Args:
            x (torch.Tensor): the input tensor
        Returns:
            outputs (torch.Tensor): the output tensor
        
        if not self.training then the forward pass is the same as the original Linear layer
        and uses the latest merged weights and biases.
        """
        outputs = super().forward(x)
        if self.training:
            if x.size(0) % self.num_heads != 0:
                raise ValueError("During training input size must be divisible by num_heads")
            outputs = outputs + self.parallel_lora_forward(x)
        return outputs

    @torch.no_grad()
    def compute_delta(self):
        """ computes the delta weight and bias for lora """
        (A, B), (b_A, b_B) = self.get_lora_params()
        return self.lora_to_delta(A, B, b_A, b_B, self.scaling)

    @torch.no_grad()
    def lora_to_delta(self, A, B, b_A, b_B, scaling):
        """ computes the delta weight and bias for lora """
        delta_weight = self.scaling * B @ A
        delta_bias = None

        if self.lora_bias:
            delta_bias = self.scaling * \
                (B @ b_A.unsqueeze(2) + b_B.unsqueeze(2)).squeeze(2)     
        return delta_weight, delta_bias

    @torch.no_grad()
    def reset_lora_parameters(self):
        """ resets lora parameters. default is orthogonal initialization """

        def init_param(params):
            for p in params:
                nn.init.orthogonal_(p)
                p.data *= math.sqrt(p.shape[1] / p.shape[0])
            return

        init_param(self.lora_A_weight.data)
        if self.lora_bias:
            nn.init.zeros_(self.lora_A_bias)

        nn.init.zeros_(self.lora_B_weight.data)    
        if self.lora_bias:
            nn.init.zeros_(self.lora_B_bias)           
        return    
