import torch.nn as nn
import lte
import torchvision.models as tvm


vit_s_32_config = {
    "patch_size": 32,
    "num_layers": 12,
    "num_heads": 6,
    "hidden_dim": 384,
    "mlp_dim": 1536,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}

model = tvm.VisionTransformer(
    image_size=224,
    num_classes=1000,
    **vit_s_32_config,
)

# NOTE: future revision will support Conv2d and LayerNorm for LoRA
only_linear = True
mode = "ddp"

# Using custom-attention because of NonDynamicallyQuantizableLinear is not LoRA compatible
lte.misc.use_custom_attention(model)

# Parameters are ignored in LTE so for exact behavior use fixed position embedding
# or modifiy the existing codebase to ensure gradients does not flow-between each other.
# We added sinusoidal embedding in lte.misc.position for vision models.
model.encoder.pos_embedding.requires_grad = False
model.class_token.requires_grad = False

if only_linear:
    ### OPTION:1
    # Converting Conv2d to Linear for simplicity (although LoRA supports Conv2d as well)
    lte.misc.replace_conv_proj_with_linear(model)

    # Disabling layer normalization affine parameters (it usually performs worse with affine parameters)
    lte.misc.disable_norm_affine_parameters(model)

    model = lte.prepare_model_for_lte(
        model.cuda(),
        lte.LTEConfig.default(
            lora_r=32,
            lora_alpha=4096,
            num_heads=32,
        ),
        mode=mode,
        strict=True,
    )
    print(model)

else:
    ### OPTION:2
    replica_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.LayerNorm))]
    replica_layers.append(model.heads)
        
    model = lte.prepare_model_for_lte(
        model.cuda(),
        lte.LTEConfig.default(
            lora_r=32,
            lora_alpha=4096,
            num_heads=32,
        ),
        replica_layers=replica_layers,
        mode=mode,
        strict=True,
    )
    print(model)
    