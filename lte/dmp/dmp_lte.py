import torch
import torch.nn as nn
from lte import LTELayer


class DistributedModelParallelLTE(LTELayer):
    """
    Virtual DMP wrapper for LTE.
    Unlike the DDP version, for N devices we create N copies of the main weight and 
    also the N LoRA parameters. Only computes forward pass once as the previous 
    weights can be directly merged into the main weight.
    This layer will automatically assign the weights to the correct device and does
    not requires torchrun.
    
    This implementation is more faithful however requires more compute and memory.
    If developing on a single node, it is recommended to proto-type on DMP 
    and re-implement the layer into DDP.
    
    Args:
        num_heads (int): number of LoRA heads
        lora_bias (bool): whether to use bias for LoRA
        lora_alpha (int): the LoRA scaling factor
        lora_r (int): the rank of LoRA    
    """
    
    def __init__(
            self, 
            num_heads, 
            lora_bias=False, 
            lora_alpha=4096,
            lora_r=32,
            ):

        self.num_heads = num_heads
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_bias = lora_bias
        self.main_device = "cuda:0"

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()     
        self.layers = nn.ModuleList()

        self.register_buffer('merged', torch.zeros(1))             
        self.register_buffer("lora_initialized", torch.zeros(1))
        return

    def convert_to_lte_module(self):
        self.scaling = self.lora_alpha / self.lora_r

        # store representation
        self._repr_A = list(self.lora_A)[0].__repr__()
        self._repr_B = list(self.lora_B)[0].__repr__()
        
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

        for p in self.layers.parameters():
            p.requires_grad_(False)
        
        self.reset_lora_parameters()
        self.lora_initialized.data[0] = 1
        return

    def __repr__(self):
        repr_str = \
            f'MultiheadLoraLayer( {self.num_heads} x ' + \
            '{\n' + \
            ' ' * 4 + 'lora_A_weight: ' + self._repr_A + '\n' + \
            ' ' * 4 + 'lora_B_weight: ' + self._repr_B + '\n' + \
            '})'
        return repr_str

    def get_lora_params(self):
        """ retrieves lora paramters """
        A = torch.stack([m.weight.to(device=self.main_device) for m in self.lora_A])
        B = torch.stack([m.weight.to(device=self.main_device) for m in self.lora_B])

        b_A, b_B = None, None
        if self.lora_bias:
            b_A = torch.stack([m.bias.to(device=self.main_device) for m in self.lora_A])
            b_B = torch.stack([m.bias.to(device=self.main_device) for m in self.lora_B])
        return (A, B), (b_A, b_B)

    @torch.no_grad()
    def merge_parameters(self):
        """ merges all lora parameters into the main module """

        def average_merging(delta_weights, delta_biases=None):
            if delta_biases is None:
                return delta_weights.mean(0), delta_biases
            return delta_weights.mean(0), delta_biases.mean(0)
        
        lora_delta_weights, lora_delta_biases = self.compute_delta()     
        
        if not self.merged:
            # register for the first time
            self.register_buffer('prev_delta_weights', torch.zeros_like(lora_delta_weights.data.clone()))

            if self.lora_bias:
                self.register_buffer('prev_delta_biases', torch.zeros_like(lora_delta_biases.data.clone()))        
        
        delta_weight, delta_bias = \
            average_merging(
                lora_delta_weights - self.prev_delta_weights, 
                lora_delta_biases if (not self.lora_bias) else \
                    lora_delta_biases - self.prev_delta_biases
        )
        
        self.prev_delta_weights.data = lora_delta_weights.data.clone()
        if self.lora_bias:
            self.prev_delta_biases.data = lora_delta_biases.data.clone()

        self.weight.data += delta_weight.data.clone().to(device=self.weight.device)
        if self.lora_bias:
            self.bias.data += delta_bias.data.clone().to(device=self.bias.device)

        for i in range(len(self.layers)):
            device = self.layers[i].weight.device
            
            self.layers[i].weight.data = self.weight.data.clone().to(device)
            self.layers[i].weight.data -= lora_delta_weights[i].data.clone().to(device)
        
            if self.lora_bias:
                self.layers[i].bias.data = self.bias.data.clone().to(device)
                self.layers[i].bias.data -= lora_delta_biases[i].data.clone().to(device)
                
        self.merged.data[0] = 1
        return delta_weight, delta_bias

    def parallel_lora_forward(self, inputs):
        """
        Applies the LoRA forward pass in parallel
        
        Args:
            inputs (Tensor): the input tensor
        Returns:
            outputs (Tensor): the output tensor
        """
        inputs = inputs.chunk(self.num_heads)
        outputs = []

        for x, lora_A, lora_B in zip(inputs, self.lora_A, self.lora_B):
            outputs.append(lora_B(lora_A(x.to(lora_A.weight.device))))

        outputs = torch.cat([x.to(device=inputs.device) for x in outputs])
        return outputs
