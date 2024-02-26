from yacs.config import CfgNode as CN


class LTEConfig():
    """
    LTE configuration.
    
    Example::
        lte_config = LTEConfig.default(lora_r=16)
        
        model = lte.deprecated.prepare_model_for_lte(
            model,
            lte_config,
        )
    """

    @staticmethod
    def default(**kwargs):
        cfg = CN()

        cfg.lora = CN()
        cfg.lora.lora_r = 8
        cfg.lora.lora_alpha = 16
        cfg.lora.lora_bias = False
        cfg.lora.num_heads = 1

        # If you want to eventually add custom layers to the model, you can add them here
        # Anything below will only be applied to Linear layers. If you want to use different 
        # LTE parameterization for differen layer, you can customize it here.
        cfg.lora.linear = CN() 
        # cfg.lora.linear.lora_r = 32

        # Override any default values with the kwargs
        LTEConfig.override_kwargs(cfg, kwargs)
        return cfg


    @staticmethod
    def override_kwargs(cfg, kwargs):
        for k, v in kwargs.items():
            if k not in cfg.lora.keys():
                raise ValueError(f"Invalid lora config {k}")
            cfg.lora[k] = v
        return cfg
