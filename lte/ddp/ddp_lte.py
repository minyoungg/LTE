import torch
import torch.nn as nn
from lte import LTELayer


class DistributedDataParallelLTE(LTELayer):
    """
    Virtual DDP wrapper for LTE.
    The main weight is shared across all LoRA devices. However, this will require 
    two forward passed on the main weight in the reset-less version. 
    This will reduce overall computation cost and memory usage. 
    Each GPU device will compute the forward pass on all the LoRA parameters. 
    This layer is meant to be used with torchrun.
    
    Use DMP for more faithful implementation.
    A more efficient implementation would be to mix both DDP and DMP. 
    
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

        self._orig_param_names = [n for n, _ in self.named_parameters()]
        [p.requires_grad_(False) for p in self.parameters()]

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()     

        self.register_buffer("lora_initialized", torch.zeros(1))
        self.register_buffer('merged', torch.zeros(1))             
        return

    def create_parallelized_params(self):
        """ creates vmap forward pass to evaluate all lora paths at the same time """      

        for param in self.parameters():
            param.requires_grad_(False)

        A_params, _ = torch.func.stack_module_state(self.lora_A)
        self.lora_A_weight = nn.Parameter(A_params['weight'])

        B_params, _ = torch.func.stack_module_state(self.lora_B)
        self.lora_B_weight = nn.Parameter(B_params['weight'])

        if self.lora_bias:
            self.lora_A_bias = nn.Parameter(A_params["bias"])
            self.lora_B_bias = nn.Parameter(B_params['bias'])
        else:
            self.lora_A_bias, self.lora_B_bias = None, None

        # skeleton template for forward pass
        self.lora_A = self.lora_A[0] 
        self.lora_B = self.lora_B[0] 

        # delete old parameters
        for m in [self.lora_A, self.lora_B]:
            for p in m._parameters.values():
                del p

        self.register_buffer("prev_lora_A_weight", torch.zeros_like(self.lora_A_weight))
        self.register_buffer("prev_lora_B_weight", torch.zeros_like(self.lora_B_weight))
        if self.lora_bias:
            self.register_buffer("prev_lora_A_bias", torch.zeros_like(self.lora_A_bias))
            self.register_buffer("prev_lora_B_bias", torch.zeros_like(self.lora_B_bias))
        else:
            self.prev_lora_A_bias, self.prev_lora_B_bias = None, None          
        return

    def convert_to_lte_module(self):
        """ converts module into LTE module """
        self.scaling = self.lora_alpha / self.lora_r

        # store representation
        self._repr_A = list(self.lora_A)[0].__repr__()
        self._repr_B = list(self.lora_B)[0].__repr__()

        self.create_parallelized_params()
        self.reset_lora_parameters()
        self.lora_initialized.data[0] = 1
        return

    def get_lora_params(self):
        """ retrieves lora paramters """
        A = self.lora_A_weight
        B = self.lora_B_weight

        b_A, b_B = None, None
        if self.lora_bias:
            b_A = self.lora_A_bias
            b_B = self.lora_B_bias
        return (A, B), (b_A, b_B)


    @torch.no_grad()
    def merge_parameters(self):
        """ merges all lora parameters into the main module """
 
        def average_merging(delta_weights, delta_biases=None):
            if delta_biases is None:
                return delta_weights.mean(0), delta_biases
            return delta_weights.mean(0), delta_biases.mean(0)
 
        lora_delta_weights, lora_delta_biases = self.compute_delta()

        if self.merged:
            # subtracting previous delta to compute correct update
            prev_delta_weights, prev_delta_biases = \
                self.lora_to_delta(
                    self.prev_lora_A_weight, 
                    self.prev_lora_B_weight, 
                    self.prev_lora_A_bias, 
                    self.prev_lora_B_bias, 
                    self.scaling
            ) 

            lora_delta_weights -= prev_delta_weights
            if self.lora_bias:
                lora_delta_biases -= prev_delta_biases

        self.prev_lora_A_weight.data = self.lora_A_weight.data.detach().clone() 
        self.prev_lora_B_weight.data = self.lora_B_weight.data.detach().clone() 

        if self.lora_bias:
            self.prev_lora_A_bias.data = self.lora_A_bias.data.detach().clone() 
            self.prev_lora_B_bias.data = self.lora_B_bias.data.detach().clone()

        delta_weight, delta_bias = average_merging(lora_delta_weights, lora_delta_biases)

        self.weight.data += delta_weight.data.clone().detach().to(self.weight.dtype)
        if self.lora_bias:
            self.bias.data += delta_bias.data.clone().detach().to(self.bias.dtype)

        self.merged.data[0] = 1
        return

    def parallel_lora_forward(self, inputs):
        """
        Chunks the inputs and applies the parallel forward pass across all LoRA layers
        For example given a batch x = [x1, x2] with 2 LoRA heads lora1 and lora2
        the output is y = [lora1(x1), lora2(x2)]. If you want same data to be processed
        across all lora heads, replicate the mini-batch by the number of heads in the 
        main optimization loop.
        
        Args:
            inputs (torch.Tensor): the input tensor of shape
        Returns:
            torch.Tensor: the output tensor of shape             
        """
        input_shape = inputs.shape

        # reshapes tensor into [num_heads x batch_size x ... x features]
        inputs = inputs.unflatten(0, (self.num_heads, -1))
        if isinstance(self, nn.Linear):
            # convert to 3D tensor [num_heads x batch_size x features]
            inputs = inputs.reshape(self.num_heads, -1, input_shape[-1])
        in_dims = 0                

        # higher chunk will be slower but can save memory
        chunk_size = 1 # math.ceil(self.num_heads // 16) 

        x = parallel_lora_forward(
            inputs,
            self.lora_A,
            self.lora_B,
            self.lora_A_weight,
            self.lora_B_weight,
            self.lora_A_bias,
            self.lora_B_bias,
            use_baddbmm_linear=isinstance(self, nn.Linear),
            in_dims=in_dims,
            chunk_size=chunk_size,
        )

        # subtract contribution of itself from previous synchronization
        x -= parallel_lora_forward(
            inputs,
            self.lora_A,
            self.lora_B,
            self.prev_lora_A_weight,
            self.prev_lora_B_weight,
            self.prev_lora_A_bias,
            self.prev_lora_B_bias,
            use_baddbmm_linear=isinstance(self, nn.Linear),
            in_dims=in_dims,
            chunk_size=chunk_size,
        )        

        # scaling parameter as a tensor
        x *= self.scaling

        if isinstance(self, nn.Linear):                
            x = x.reshape(*input_shape[:-1], x.shape[-1])
        else:
            x = x.flatten(0, 1)
        return x


def baddbmm_linear(x, lora_A_weight, lora_A_bias, lora_B_weight, lora_B_bias):
    """ 
    Batched matmul using BLAS and LAPACK operations.
    Faster than vmapping using.
    
    Args:
        x (torch.Tensor): input tensor of shape [num_heads x batch_size x features]
        loar_A_weight (torch.Tensor): first LoRA parameters of shape [num_heads x in_features x r]
        loar_A_bias (torch.Tensor): first LoRA bias of shape [num_heads x r]
        loar_B_weight (torch.Tensor): second LoRA parameters of shape [num_heads x r x out_features]
        loar_B_bias (torch.Tensor): second LoRA bias of shape [num_heads x out_features]
    Returns:
        torch.Tensor: output tensor of shape [num_heads x batch_size x out_features]
    
    NOTE: always assumes sequence dimension is flattened with the batch dimension.
    """
    assert len(x.shape) == 3, f'Expected 3D tensor got {x.shape}'
    
    if lora_A_bias is not None:
        x = torch.baddbmm(lora_A_bias.unsqueeze(1), x, lora_A_weight.permute(0, 2, 1))
        x = torch.baddbmm(lora_B_bias.unsqueeze(1), x, lora_B_weight.permute(0, 2, 1))
    else:
        x = torch.bmm(x, lora_A_weight.permute(0, 2, 1))
        x = torch.bmm(x, lora_B_weight.permute(0, 2, 1))
    return x


def mhlora_baddbmm_linear(x, lora_A_weight, lora_A_bias, lora_B_weight, lora_B_bias):
    """ 
    Special case of baddbmm_linear to reduce memory usage.
    The same input is used for all heads. 
    Good for MHLoRA or evaluation while still enabling LTE.
    
    Args:
        x (torch.Tensor): input tensor of shape [batch_size x features]
        loar_A_weight (torch.Tensor): first LoRA parameters of shape [num_heads x in_features x r]
        loar_A_bias (torch.Tensor): first LoRA bias of shape [num_heads x r]
        loar_B_weight (torch.Tensor): second LoRA parameters of shape [num_heads x r x out_features]
        loar_B_bias (torch.Tensor): second LoRA bias of shape [num_heads x out_features]
    Returns:
        torch.Tensor: output tensor of shape [num_heads x batch_size x out_features]
        
    NOTE: for cleanliness of the code, we disabled force paralleization option and 
    this function not used in the current codebase, however we leave it for those who might want to use it.
    """
    assert len(x.shape) == 2, f'Expected 2D tensor got {x.shape}'
    
    (b, f_in), (h, f_out) = x.shape, lora_B_weight.shape[:2]

    # more memory efficient than tiling it first and using baddbmm
    x = (x @ lora_A_weight.view(-1, f_in).T).unflatten(-1, (h, -1))
    if lora_A_bias is not None:
        x += lora_A_bias.unsqueeze(0)

    x = x.swapaxes(0, 1)
    if lora_B_bias is not None:
        x = torch.baddbmm(lora_B_bias.unsqueeze(1), x, lora_B_weight.transpose(1, 2))
    else:
        x = torch.bmm(x, lora_B_weight.transpose(1, 2))
    return x.view(h, b, f_out)


def parallel_lora_forward(
    x, 
    lora_A_fn, 
    lora_B_fn, 
    lora_A_weight, 
    lora_B_weight, 
    lora_A_bias=None, 
    lora_B_bias=None, 
    use_baddbmm_linear=False, 
    in_dims=(0, 0, None), 
    chunk_size=1
):
    """
    Applies the forward pass of the parallel Lora network.

    Args:
        x (torch.Tensor): Input tensor. 
        lora_A_fn (callable): Function that applies the forward pass of the first Lora network.
        lora_B_fn (callable): Function that applies the forward pass of the second Lora network.
        lora_A_weight (torch.Tensor): Weight tensor for the first Lora network.
        lora_B_weight (torch.Tensor): Weight tensor for the second Lora network.
        lora_A_bias (torch.Tensor, optional): Bias tensor for the first Lora network. Defaults to None.
        lora_B_bias (torch.Tensor, optional): Bias tensor for the second Lora network. Defaults to None.
        use_baddbmm_linear (bool, optional): If True, uses baddbmm for faster computation. Defaults to False.
        in_dims (tuple, optional): Tuple of input dimensions for torch.vmap. Defaults to (0, 0, None).
        chunk_size (int, optional): Chunk size for torch.vmap. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after applying the forward pass of the parallel Lora network.
    """
    if use_baddbmm_linear:
        if len(x.shape) == 2:
            baddbmm_function = mhlora_baddbmm_linear
        elif len(x.shape) == 3:
            baddbmm_function = baddbmm_linear
        else:
            raise RuntimeError(f'Unsupported input shape {x.shape}')
            
        x = baddbmm_function(
                x, 
                lora_A_weight, 
                lora_A_bias, 
                lora_B_weight, 
                lora_B_bias, 
                )
    else:
            
        def parallelize_A(params, buffers, data):
            return torch.func.functional_call(lora_A_fn, (params, buffers), (data,))

        def parallelize_B(params, buffers, data):
            return torch.func.functional_call(lora_B_fn, (params, buffers), (data,))
                
        vmap_A = torch.vmap(parallelize_A, in_dims=in_dims, chunk_size=chunk_size)
        vmap_B = torch.vmap(parallelize_B, chunk_size=chunk_size)
                
        if lora_A_bias is not None:
            x = vmap_A({'weight': lora_A_weight, 'bias': lora_A_bias}, {}, x)
        else:
            x = vmap_A({'weight': lora_A_weight}, {}, x)        
        
        if lora_B_bias is not None:
            x = vmap_B({'weight': lora_B_weight, 'bias': lora_B_bias}, {}, x)        
        else:
            x = vmap_B({'weight': lora_B_weight}, {}, x)       
    return x
