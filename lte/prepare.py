import numpy as np
import torch.nn as nn
import warnings
from lte.replica import MultiheadReplicaLayer


def prepare_model_for_lte(
        model, 
        lora_config, 
        copy_weights=True, 
        strict=True, 
        replica_layers=[],
        mode="ddp",
        ):
    """
    Convert a model into an LTE model.
    
    Args:
        model (nn.Module): the model to convert
        lora_config (LTEConfigs): the lora config to use
        copy_weights (bool): if True, copy weights from the original model to the new model
        strict (bool): if True, raise an error if not all parameters are converted
        replica_layers (list): list of modules to convert for standard local-step averaging.
        mode (str): the mode to use. Options are "ddp" and "dmp"
        
    Returns:
        model (nn.Module): LTE model
    
    Example::
        model = \
            lte.prepare_model_for_lte(
                model,
                lte.misc.LTEConfig.default(
                    lora_r=16,
                    num_heads=4,
                )
        )
    """
        
    if mode == "ddp":
        from lte.ddp.linear import MultiheadLoRALinear
    elif mode == "dmp":
        from lte.dmp.linear import MultiheadLoRALinear
        assert next(model.parameters()).is_cuda, "dmp expects model to be in cuda"
    elif mode == 'mhlora': 
        from lte.mhlora.linear import MultiheadLoRALinear
    else:
        raise ValueError(f"mode {mode} not recognized")
        

    lora_kwargs = lora_config.lora
    linear_lora_kwargs = patch_kwargs(lora_kwargs, lora_kwargs.linear)
    orig_linear_lora_alpha = linear_lora_kwargs['lora_alpha']

    # replace pytorch attention with custom attention since we look for nn.Linear
    converted_parameter_count = 0
    trainable_parameter_count = sum([p.numel() for p in model.parameters() if p.requires_grad])

    supported_modules = (
        nn.Linear,
        # nn.LayerNorm,
        # nn.Conv2d,
        # nn.Embedding,
    )
    
    for n, m in model.named_modules():

        if not isinstance(m, supported_modules) and (not (m in replica_layers)):
            continue
        
        if np.any([is_submodule(rm, m) for rm in replica_layers]):
            continue

        parent_module, old_module, target_name = _get_submodules(model, n)
        converted_parameter_count += sum([p.numel() for p in old_module.parameters() if p.requires_grad])

        dtype = next(old_module.parameters()).dtype
                
        if m in replica_layers:
            new_module = MultiheadReplicaLayer(
                old_module,
                num_heads=lora_kwargs.num_heads,
                mode=mode,
            ).to(dtype=dtype)
        else:
            if isinstance(m, nn.Linear):
                device = next(old_module.parameters()).device
                                
                new_module = MultiheadLoRALinear(
                    old_module.in_features,
                    old_module.out_features,
                    bias=(old_module.bias is not None),                    
                    **linear_lora_kwargs,
                ).to(device=device, dtype=dtype)

                if copy_weights:
                    if mode == 'ddp':
                        new_module.weight.data = old_module.weight.data
                        if old_module.bias is not None:
                            new_module.bias.data = old_module.bias.data
                    else:
                        new_module.weight.data = old_module.weight.data.clone().to(new_module.weight.device)
                        if old_module.bias is not None:
                            new_module.bias.data = old_module.bias.data.clone().to(new_module.bias.device)
                        
                        if mode == 'dmp':
                            # dmp creates N copies of the original weight so it requires further copying
                            for l in new_module.layers:
                                l.weight.data = old_module.weight.data.clone().to(l.weight.device)

                                if old_module.bias is not None:
                                    l.bias.data = old_module.bias.data.clone().to(l.bias.device)
                           
            else:
                print("module replacement rule not found")

        setattr(parent_module, target_name, new_module)

    if converted_parameter_count != trainable_parameter_count:
        diff = trainable_parameter_count - converted_parameter_count
        e_msg = f"Converted parameter count {converted_parameter_count} " + \
                f"does not match trainable parameter count {trainable_parameter_count} [diff: {diff}]"
        if strict:
            raise RuntimeError(e_msg)
        else:
            warnings.warn(e_msg)
    return model

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def patch_kwargs(kwargs, new_kwargs):
    kwargs = kwargs.copy()
    for key, value in new_kwargs.items():
        kwargs[key] = value

    for k, v in list(kwargs.items()):
        if isinstance(v, dict):
            del kwargs[k]
    return kwargs

def is_submodule(parent_module, submodule):
    return np.any([mod is submodule for mod in parent_module.modules()]) and (parent_module is not submodule)
