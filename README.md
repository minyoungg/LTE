# LTE: LoRA-the-Explorer

<a href="https://minyoungg.github.io/lte/">LoRA-the-explorer (LTE)</a> is a framework to fine-tune and pre-train models without directly optimizing over the main weights.
This is a minimal re-implementation of the codebase with tools for small- to mid-scale research development.


## Installation
Tested on Ubuntu with Python 3.11 and PyTorch2.1/2.2. Older torch versions may not support some operators used in the codebase.

```bash
git clone https://github.com/minyoungg/LTE
cd LTE
pip install -e .
```

## Example usage
By default, this codebase uses the reset-less version (see Appendix in the paper)

```python
import lte

# your neural network
model = MyModel()

# converts into an LTE model
model = lte.prepare_model_for_lte(
      model.cuda(),
      lte.LTEConfig.default(
          lora_r=32,
          lora_alpha=4096,
          num_heads=32,
      ),
)
```

Given a mini-batch, LTE will automatically chunk up the batch size and parallelize it across each LoRA head.

```python
x = get_data()
assert x.size(0) % 32 == 0, 'make sure batch-size is divisible by num_heads'
model(x)
```

To merge the model, you can use a merge scheduler `lte.misc.merge.MergeCondition`, or you can implement your own. For example:

```python
for n, m in model.named_modules():
    if isinstance(m, lte.LTELayer, lte.ReplicaLayer):
        m.merge_parameters()
```

If you have layers that are not supported, you can pass them as a replica layer, which will replicate the layer across all devices. These parameters are averaged when merged. Unfortunately, replica layers will likely require a separate learning rate from the LoRA parameters.

```python
model = lte.prepare_model_for_lte(
      model.cuda(),
      lte.LTEConfig.default(
          lora_r=32,
          lora_alpha=4096,
          num_heads=32,
      ),
      replica_layers=[model.ignore_this_layer]
)
```

We include some helpful functions that might be useful.

```python
# convert Conv2D projection layers in ViT with their linear counterparts
lte.misc.replace_conv_proj_with_linear(model)

# disables affine parameters in LayerNorm
# from my experience, disabling results in better performance for both LTE and standard training
lte.misc.disable_norm_affine_parameters(model)
```

The current codebase was mainly used for artificially emulating distributed training pipelines.
We currently provide `DistributedDataParallel`(DDP) and `DistributedModelParallel`(DMP). (A better name could have been chosen).

Here is a quick TLDR of how they are implemented.
Assume we have `H` virtual-devices on `N` gpu-devices. In DDP mode, we will create `1` main weight and `H` LoRA parameters. 
The LoRA devices will share the forward pass of the main weight across all `H` virtual devices since it is redundant.
Using `torchrun` will chunk the data across devices and also across virtual devices.
This will keep memory and compute costs low for development purposes. 

DMP mode is more faithful to how it will be implemented in practice. DMP creates `H` copies of the main weights and `H` LoRA parameters distributed across `N` devices.
Most PyTorch cuda operations should be non-blocking, but they will still run much slower than DMP as they do not share any computation between the virtual devices.

You can switch between these modes via a flag.
```python
model = lte.prepare_model_for_lte(
      ...
      mode='ddp' # or 'dmp' ('ddp' by default)
)
```

DMP will automatically distribute across all visible cuda-devices without using `torchrun`, so make sure you set the visibility correctly.
```bash
# will automatically distribute across 4 devices
CUDA_VISIBLE_DEVICES=1,2,3,4 python lte_dmp_train_script.sh
```

DDP should be used with `torchrun`.  

### Helpful guidelines
- First, test whether the mhlora parameterization of the model will converge to the same test loss. We added `mode="mhlora"` to help you with this.
- Note that different alpha values might result in the same training loss, but vastly different test loss. Alpha values of (1024, 2048, 4096, 8192) is a good range to search over.
- If mhlora matches the pre-training performance, LTE with `merge_iter=1` can recover the same performance. 
- LTE will require longer training iteration to converge since the mini-batch is sharded across each head. Using a larger batch size may help.
- Next, increase the `merge_iter` to get the asynchronous benefits.


### MORE CODE COMING SOON 
- [ ] 4bit quantization support
- [ ] Layernorm and Conv2d support
- [ ] Full training example

Note: we do not support standalone parameters, so wrap it as a module to replicate.

### Citation
If you found the library useful, please consider citing
```bibtex
@article{huh2024lte,
  title={Training Neural Networks from Scratch with Parallel Low-Rank Adapters},
  author={Huh, Minyoung and Cheung, Brian and Bernstein, Jeremy and Agrawal, Pulkit and Isola, Phillip},
  journal={arXiv preprint arXiv},
}
```
