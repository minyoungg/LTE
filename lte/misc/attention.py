import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lte.misc import common


class MultiheadAttention(nn.Module):
    """ MultiHead Attention using PyTorch's scaled_dot_product_attention """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        bias=True,
        split_qkv=True,
    ):
        super().__init__()
        self.bias = bias
        self.heads = num_heads
        self.dropout = dropout
        self.split_qkv = split_qkv

        if self.split_qkv:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.init_weights()
        return

    def init_weights(self):
        """
        Using same initialization protocol for PyTorch's MultiheadAttention
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1041
        """
        if self.split_qkv:
            for m in [self.q_proj, self.k_proj, self.v_proj]:
                torch.nn.init.xavier_uniform_(m.weight)
                if self.bias:
                    torch.nn.init.constant_(m.bias, 0.0)
        else:
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            if self.bias:
                torch.nn.init.constant_(self.in_proj.bias, 0.0)

        if self.bias:
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
        return

    def in_projection(self, q, k, v):
        """
        Args:
            q, k, v: torch.Tensor of shape (B, S, D)
        Returns:
            q, k, v: torch.Tensor of shape (B, H, S, D_head)
        """
        if self.split_qkv:
            q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        else:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
       
        q, k, v = (
            q.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            k.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            v.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
        )
        return q, k, v

    def forward(self, q, k, v, need_weights=False):
        q, k, v = self.in_projection(q, k, v)
        assert need_weights == False, "need_weights is not supported in this version"
        
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout,
        ).permute(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(out), None


class DeprecatedMultiheadAttention(nn.Module):
    """
    This version is deprecated and will be removed in the future. Please use MultiheadAttention instead.
    PyTorch 2.2 now natively supports flash-attention-2
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        bias=True,
        split_qkv=True,
    ):
        print("This version of MultiheadAttention is deprecated and will be removed in the future.")
        super().__init__()
        self.bias = bias
        self.heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.split_qkv = split_qkv
        self.flash_available, self.flash_attn = self.check_flash_available()

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if self.split_qkv:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.init_weights()
        return
    
    def check_flash_available(self):
        try:
            if common.flash_ready_device():
                from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

                flash_available = True
            else:
                flash_available = False
                print("The current device does not support flash-attention.")
        except ImportError:
            flash_available = False
            flash_attn_qkvpacked_func = None
            print(
                "Flash-attention not available. "
                + "Please install it from https://github.com/Dao-AILab/flash-attention"
            )
        return flash_available, flash_attn_qkvpacked_func


    def init_weights(self):
        """
        Using same initialization protocol as pytorch
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1041
        """
        if self.split_qkv:
            for m in [self.q_proj, self.k_proj, self.v_proj]:
                torch.nn.init.xavier_uniform_(m.weight)
                if self.bias:
                    torch.nn.init.constant_(m.bias, 0.0)
        else:
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            if self.bias:
                torch.nn.init.constant_(self.in_proj.bias, 0.0)

        if self.bias:
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
        return

    def in_projection(self, q, k, v):
        if self.split_qkv:
            q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
            q, k, v = (
                q.unflatten(-1, (self.heads, -1)),
                k.unflatten(-1, (self.heads, -1)),
                v.unflatten(-1, (self.heads, -1)),
            )
            if not self.flash_available:
                q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d"), [q, k, v])
        else:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
            if not self.flash_available:
                q, k, v = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
                    [q, k, v],
                )
        return q, k, v

    def forward(self, q, k, v, need_weights=True):
        q, k, v = self.in_projection(q, k, v)
        
        if self.flash_available:
            # qkv in float16 or bfloat16
            qkv = torch.stack([q, k, v], dim=2)
            qkv = common.autocast_vars(qkv)

            out = self.flash_attn(
                qkv=qkv,
                dropout_p=0.0,
                softmax_scale=self.scale,
                return_attn_probs=True,
            )
            out, attn, _ = out
            out = out.flatten(-2, -1)
        else:
            q, k, v = common.autocast_vars(q, k, v)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

        if not need_weights:
            attn = None

        return self.dropout(self.out_proj(out)), attn
