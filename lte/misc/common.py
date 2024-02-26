import torch
from contextlib import nullcontext


def determine_compute_context():
    """
    Automatically determines the compute dtype and context manager.
    Override this function if you want to use a different dtype or context manager.
    
    Args:
        None
    Returns:
        dtype: torch.dtype for the model parameters not AMP dtype
        context: torch.cuda.amp.Gradscaler or contextlib.contextmanager 
        scaler: torch.cuda.amp.GradScaler
        
    NOTE: it seems like even if we use bfloat16, using high-precision is important for 
    numerical stability of the merge. Without using AMP sequential merigng hurts performance.
    In practice one would keep a high-precision copy regardless, which was removed for simplicity.
    Until the feature for high-precsision parameters with quantization is re-implemented
    We will use AMP with bfloat16 also.
    """

    if torch.cuda.is_bf16_supported():
        # dtype = torch.bfloat16
        dtype = torch.float32
        # context = nullcontext()
        # scaler = None
        context = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
    else:
        dtype = torch.float32
        context = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()

    return dtype, context, scaler


def auto_dtype():
    """ Determines the compute dtype depending on the device. """
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_vars(*vars):
    """
    This function was needed when using AMP with Dao-AILab/flash-attention.
    As it does not auto-cast if some of the layers are float32 (e.g. LN).
    Since the native FA2 support from PyTorch 2.2 this function is no longer needed.
    """
    def _autocast(x):
        if torch.is_autocast_enabled() and x.dtype != auto_dtype():
            print('forced autocasting')
            return x.to(dtype=auto_dtype())
        return x

    if len(vars) == 1:
        return _autocast(vars[0])
    return (*[_autocast(x) for x in vars],)


def flash_ready_device():
    """ Check if the device has support flash-attention. """
    if not torch.cuda.is_available():
        return False

    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    return major >= 8
