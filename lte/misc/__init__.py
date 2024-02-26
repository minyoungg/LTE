import gc
import torch
import torch.nn as nn
import einops

from lte.misc.attention import MultiheadAttention
from lte.misc import merge, position, common


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def use_custom_attention(model, split_qkv=False):
    """
    Replaces torch.MultiheadAttention with custom MultiheadAttention. 
    LTE looks for nn.Linear which is not used in the torch.MultiheadAttention.
    Updates model in place but returns the model for convenience.
    
    Args:
        model (nn.Module): the model to convert
    Returns:
        model (nn.Module): the model with custom MultiheadAttention modules
    
    Example::
        model = lte.misc.use_custom_attention(model)
    """

    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        parent_module, old_module, target_name = _get_submodules(model, key)

        if isinstance(old_module, nn.MultiheadAttention):
            new_module = MultiheadAttention(
                embed_dim=old_module.embed_dim,
                num_heads=old_module.num_heads,
                dropout=old_module.dropout,
                bias=(old_module.in_proj_bias is not None),
                split_qkv=split_qkv,
            )

            if not new_module.split_qkv:
                new_module.in_proj.weight.data = old_module.in_proj_weight.data
                new_module.in_proj.bias.data = old_module.in_proj_bias.data
                new_module.out_proj.weight.data = old_module.out_proj.weight.data
                new_module.out_proj.bias.data = old_module.out_proj.bias.data

            setattr(parent_module, target_name, new_module)
            del old_module

    torch.cuda.empty_cache()
    gc.collect()
    return model


class LinearProjection(nn.Module):
    """
    Linear projection layer
    
    Args:
        hidden_dim (int): the hidden dimension of the linear projection
        patch_size (int): the patch size of the input image
    """
    def __init__(self, hidden_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(
            3 * patch_size * patch_size, hidden_dim, bias=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): the input tensor of shape (b, c, h, w)
        Returns:
            x (torch.Tensor): the output tensor of shape (b, embed, h//patch_size, w//patch_size)
        """
        _, _, h, w = x.shape
        # patchify
        x = einops.rearrange(
            x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.linear_proj(x)

        # reshape to the desired output shape
        x = einops.rearrange(
            x, 'b (h w) c -> b c h w',
            h=h//self.patch_size,
            w=w//self.patch_size,
        )
        return x


def replace_conv_proj_with_linear(model):
    """
    Replaces all Conv2d modules with kernel_size == stride with LinearProjection.
    This is useful for replacing the first layer of a vision transformer with a linear projection.
    Helpful for using LoRA on ViT Conv2D projection.
    Updates model in place but returns the model for convenience.

    Args:
        model (nn.Module): the model to convert
    Returns:
        model (nn.Module): the model with replaced conv2d modules
    
    Example::
        model = lte.misc.replace_conv_proj_with_linear(model)
    """
    for k, m in model.named_modules():
        parent_module, old_module, target_name = _get_submodules(model, k)
        
        # replace all conv2d that have same kernel_size and stride with linear projection    
        if isinstance(old_module, nn.Conv2d) and old_module.kernel_size == old_module.stride:
            new_module = LinearProjection(old_module.out_channels, old_module.kernel_size[0])
            
            new_module.linear_proj.weight.data.copy_(
                old_module.weight.data.moveaxis(1, -1).reshape(old_module.out_channels, -1).clone()
            )
            new_module.linear_proj.bias.data.copy_(
                old_module.bias.data.clone()
            )
            
            setattr(parent_module, target_name, new_module)
            del old_module
    return model


def disable_norm_affine_parameters(model):
    """
    Disables the affine parameters of all LayerNorm and BatchNorm2d modules in the model.
    Updates model in place but returns the model for convenience.
    
    Args:
        model (nn.Module): the model to disable the affine parameters of
    Returns:
        model (nn.Module): the model with disabled affine parameters
        
    Example::
        model = lte.misc.disable_norm_affine_parameters(model)

    NOTE: Feel free to add other normalization layers as needed. 
    """
    for n, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            m.weight = None
            m.bias = None
            m.elementwise_affine = False
        elif isinstance(m, nn.BatchNorm2d):
            m.weight = None
            m.bias = None
            m.affine = False
    return model
