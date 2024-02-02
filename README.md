# RTM-BNN

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

## CUDA-based Error Model Application and Binarization/Quantization

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

Here is a list of the command line parameters for running the error evaluations with SPICE-Torch:
| Command line parameter | Options |
| :------------- |:-------------|
| --model      | FC, VGG3, VGG7, ResNet |
| --dataset      | MNIST, FMNIST, QMNIST, SVHN, CIFAR10, IMAGENETTE |
| --an-sim      | int, whether to turn on the mapping from SPICE, default: None |
| --mapping      | string, loads a direct mapping from the specified path, default: None |
| --mapping-distr      | string, loads a distribution-based mapping from the specified path, default: None |
| --array-size      | int, specifies the size of the crossbar array, default: None |
| --performance-mode      | int, specify whether to activate the faster and more memory-efficient performance mode (when using this sub-MAC results can only be changed in cuda-kernel!), default: None |
| --print-accuracy      | int, specifies whether to print inference accuracy, default: None |
| --test-error-distr      | int, specifies the number of repetitions to perform in accuracy evaluations for distribution based evaluation, default: None |
| --train-model      | bool, whether to train a model, default: None|
| --epochs      | int, number of epochs to train, default: 10|
| --lr      | float, learning rate, default: 1.0|
| --gamma      | float, learning rate step, default: 0.5|
| --step-size      | int, learning rate step site, default: 5|
| --batch-size      | int, specifies the batch size in training, default: 64|
| --test-batch-size      | int, specifies the batch size in testing, default: 1000|
| --save-model | string, saves a trained model with the specified name in the string, default:None |
| --load-model-path | string, loads a model from the specified path in the string, default: None |
| --load-training-state | string, saves a training state with the specified name in the string, default:None |
| --save-training-state | string, loads a training state from the specified path in the string, default: None |
| --gpu-num | int, specifies the GPU on which the training should be performed, default: 0 |
| --profile-time | int, Specify whether to profile the execution time by specifying the repetitions, default: None |
| --extract-absfreq | int, Specify whether to extract the absolute frequency of MAC values, default: None |
| --extract-absfreq-resnet | int, Specify whether to extract the absolute frequency of MAC values for resnet, default: None |
