# RTM-BNN

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

To run

``` bash ./run_auto_all.sh NN_MODEL BLOCK_SIZE LOOPS LAYER GPU ```

where 

```
NN_MODEL:    RESNET (for CIFAR check other repo)
BLOCK_SIZE:  nanowire size: 64, 32, 16, 8, 4, 2
LOOPS:       number of inference steps (usually 1 or 100)
LAYER:       which layer should be unprotected (individual: from 0-20) or ALL (at once) or CUSTOM (needs to set 0 manually in run_auto_all.sh at declare -a PROTECT_LAYERS=(...) )
GPU:         which GPU id to use
```

To change `error_rate` go into `run_auto_all.sh` and choose or edit around line 100 `declare -a PERRORS=(0.0000455)`

example for RESNET loops=1 block_size=64 layer=6 (index in cli is from 0) gpu=0:

``` bash ./run_auto_all.sh RESNET 1 64 5 0 ```

# SPICE-Torch
A framework for connecting SPICE simulations of analog computing neuron circuits with PyTorch accuracy evaluations for Binarized (and soon Quantized) Neural Networks.
