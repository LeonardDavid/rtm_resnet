import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

import custommac1d
import custommac2d
import mappingdirect
import mappingdistr
import custommac1dmappingdirect
import custommac2dmappingdirect
import custommac1dmappingdistr
import custommac2dmappingdistr

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):
        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

quantize = Quantize.apply


class ErrorModel(Function):
    @staticmethod
    def forward(ctx, input, index_offset, block_size = 64, error_model=None):
        output = input.clone().detach()
        output = error_model.applyErrorModel(input=output, index_offset=index_offset, block_size=block_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

apply_error_model = ErrorModel.apply

# add for compatibility to every apply_error_model parameters that do not use index_offset and block_size
index_offset_default = np.zeros([2,2])
block_size_default = 1.0

def check_quantization(quantize_train, quantize_eval, training):
    condition = ((quantize_train == True) and (training == True)) or ((quantize_eval == True) and (training == False)) or ((quantize_train == True) and (quantize_eval == True))

    if (condition == True):
        return True
    else:
        return False


class QuantizedActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedActivation"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.training = None
        super(QuantizedActivation, self).__init__(*args, **kwargs)

    def forward(self, input):
        output = None
        check_q = check_quantization(self.quantize_train,
         self.quantize_eval, self.training)
        if (check_q == True):
            output = quantize(input, self.quantization)
        else:
            output = input
        if self.error_model is not None:
            output = apply_error_model(output, self.error_model)
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.an_sim = kwargs.pop('an_sim', None)
        self.array_size = kwargs.pop('array_size', None)
        self.mapping = kwargs.pop('mac_mapping', None)
        self.mapping_distr = kwargs.pop('mac_mapping_distr', None)
        self.sorted_mapping_idx = kwargs.pop('sorted_mac_mapping_idx', None)
        self.performance_mode = kwargs.pop('performance_mode', None)
        self.training = kwargs.pop('train_model', None)
        self.extract_absfreq = kwargs.pop('extract_absfreq', None)
        if self.extract_absfreq is not None:
            self.absfreq = torch.zeros(self.array_size+1, dtype=int).cuda()
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.lost_vals_r = kwargs.pop('lost_vals_r', None)
        self.lost_vals_l = kwargs.pop('lost_vals_l', None)
        self.block_size = kwargs.pop('block_size', None)
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight
            if self.error_model is not None:

                if self.test_rtm is not None:
                    shift = 0
                    nr = 0
                    for i in range(0, self.index_offset.shape[0]):
                        for j in range(0, self.index_offset.shape[1]):
                            nr += 1
                            if(random.uniform(0.0, 1.0) < self.error_model.p):
                                shift += 1
                                # 50/50 random possibility of right or left shift
                                if(random.choice([-1,1]) == 1):
                                    # right shift
                                    self.index_offset[i][j] += 1
                                    if (self.index_offset[i][j] > 64/2):
                                        self.lost_vals_r[i][j] += 1
                                        quantized_weight[i][(j+1)*64 - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                    if (self.lost_vals_l[i][j] > 0):
                                        self.lost_vals_l[i][j] -= 1
                                else:
                                    # left shift
                                    self.index_offset[i][j] -= 1
                                    if(-self.index_offset[i][j] > 64/2):
                                        self.lost_vals_l[i][j] += 1
                                        quantized_weight[i][j*64 + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                    if(self.lost_vals_r[i][j] > 0):
                                        self.lost_vals_r[i][j] -= 1

                    # print("(lin) total shifts: " + str(shift) + "/" + str(nr))
                    
                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)

                    # print(np.sum(self.lost_vals_r))
                    # print(np.sum(self.lost_vals_l))
                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.block_size, self.error_model)
            if self.an_sim is not None:
                # compute weight and input shapes
                wm_row = quantized_weight.shape[0]
                wm_col = quantized_weight.shape[1]
                im_col = input.shape[0]
                # assign weights and inputs
                weight_b = quantized_weight
                input_b = input
                # size for output buffer
                if self.performance_mode is not None:
                    # print("performance mode 1d")
                    # print("performance mode", self.performance_mode)
                    output_b = torch.zeros(im_col, wm_row).cuda()

                    if self.mapping is not None:
                        # print("mapping in performance mode")
                        custommac1dmappingdirect.custommac1dmappingdirect(input_b, weight_b, output_b, self.mapping, self.array_size)
                        output_b = output_b.detach()

                    if self.mapping_distr is not None:
                        custommac1dmappingdistr.custommac1dmappingdistr(input_b, weight_b, output_b, self.mapping_distr, self.sorted_mapping_idx, self.array_size)
                        output_b = output_b.detach()
                    # print("custommac1d")
                    ## check correctness
                    # correct = torch.eq(output_b, output)
                    # correct = (~correct).sum().item()
                    # # 0 if tensors match
                    # print("correctness: ", correct)
                    if self.training is not None:
                        output = F.linear(input, quantized_weight)
                        output.data.copy_(output_b.data)
                    else:
                        output = output_b
                else:
                    buffer_size = int(np.ceil(wm_col/self.array_size))
                    output_b = torch.zeros(im_col, wm_row, buffer_size).cuda()
                    # print("im_col", im_col)
                    # print("bs", buffer_size)
                    # print("array_size", self.array_size)
                    # print("b", output_b.shape)
                    # call custom mac
                    custommac1d.custommac1d(input_b, weight_b, output_b, self.array_size)
                    # print("direct")

                    if self.extract_absfreq is not None:
                        # get popcount value (unsigned int, max at self.array_size)
                        # print("1", output_b)
                        output_b_pop = (output_b + self.array_size)/2
                        # print("pop", output_b_pop)
                        output_int = torch.flatten(output_b_pop.int())
                        # print("out int", output_int)
                        self.absfreq += torch.bincount(output_int, minlength=self.array_size+1)
                        # print("absfreq", self.absfreq)
                        # print("pop2", output_b_pop)
                        # transform back to format that is needed by pytorch
                        output_b = 2*output_b_pop - self.array_size
                        # print("2", output_b)

                    # apply mapping
                    if self.mapping is not None:
                        # get popcount value (unsigned int, max at self.array_size)
                        # print("1", output_b)
                        output_b_pop = (output_b + self.array_size)/2
                        # print("pop", output_b_pop)
                        mappingdirect.mappingdirect(output_b_pop, self.mapping)
                        # print("pop2", output_b_pop)
                        # transform back to format that is needed by pytorch
                        output_b = 2*output_b_pop - self.array_size
                        # print("2", output_b)

                    if self.mapping_distr is not None:
                        # print("in mapping_distr")
                        # create buffer for accumulating random values
                        output_b_pop = (output_b + self.array_size)/2
                        mappingdistr.mappingdistr(output_b_pop, self.mapping_distr, self.sorted_mapping_idx)
                        output_b = 2*output_b_pop - self.array_size

                    # mappingdirect.mappingdirect(output_b, self.mapping)
                    # [-32-] [-32-] ... [-5-] #
                    # clock freq: 1ns (also in SPICE)
                    # [-ift-] [-ift-] ... [ift] ## 7.8 ns in SPICE
                    # [-cft-] [-cft-] ... [cft] ## 10 ns in SPICE
                    # [-amac-] [-amac-] ... [-amac-] ## 1/10
                    # 1) nominal input var
                    # 2) input with process variation input
                    # popcount 0
                    # sum
                    # MAPPING: Popcountcount value -> Zyklen -> Approximate MAC Ergebnis
                    # [-1-] # user can use their own mapping popcount -> approx. mac
                    # detach, so that computations are not considered during training
                    # print(output_b.shape)
                    ### --- apply error model to output_b
                    ### --- apply snn simulation
                    ###
                    # print("pop scale", output_b_pop)
                    # normal distribution
                    output_b = torch.sum(output_b, 2)
                    output_b = output_b.detach()
                    # execute standard way, to create computation graph for backprop
                    output = F.linear(input, quantized_weight)
                    # replace custom data with standard data, without touching computation graph
                    output.data.copy_(output_b.data)
                    # output = output_b
                    # print("custommac1d")
                    # check correctness
                    # correct = torch.eq(output_b, output)
                    # correct = torch.isclose(output_b, output, atol=1e-3)
                    # correct = (~correct).sum().item()
                    # # 0 if tensors match
                    # print("correctness: ", correct)
                    # print("out_b", output_b)
                    # print("out", output)
            else:
                output = F.linear(input, quantized_weight)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.an_sim = kwargs.pop('an_sim', None)
        self.array_size = kwargs.pop('array_size', None)
        self.mapping = kwargs.pop('mac_mapping', None)
        self.mapping_distr = kwargs.pop('mac_mapping_distr', None)
        self.sorted_mapping_idx = kwargs.pop('sorted_mac_mapping_idx', None)
        self.performance_mode = kwargs.pop('performance_mode', None)
        self.training = kwargs.pop('train_model', None)
        self.extract_absfreq = kwargs.pop('extract_absfreq', None)
        if self.extract_absfreq is not None:
            self.absfreq = torch.zeros(self.array_size+1, dtype=int).cuda()
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.lost_vals_r = kwargs.pop('lost_vals_r', None)
        self.lost_vals_l = kwargs.pop('lost_vals_l', None)
        self.block_size = kwargs.pop('block_size', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                # weight_b1 = quantized_weight.view(self.out_channels,-1).cuda()
                # wm = weight_b1.shape
                # input_b1 = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride).cuda()
                # im = input_b1.shape
                # print("wm, im", wm, im)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:

                if self.test_rtm is not None:
                    # print(self.index_offset)
                    shift = 0
                    nr = 0
                    for i in range(0, self.index_offset.shape[0]):
                        for j in range(0, self.index_offset.shape[1]):
                            nr += 1
                            if(random.uniform(0.0, 1.0) < self.error_model.p):
                                shift += 1
                                # 50/50 random possibility of right or left shift
                                if(random.choice([-1,1]) == 1):
                                    # right shift
                                    self.index_offset[i][j] += 1
                                    if (self.index_offset[i][j] > 64/2):
                                        self.lost_vals_r[i][j] += 1
                                        quantized_weight[i][(j+1)*64 - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                    if (self.lost_vals_l[i][j] > 0):
                                        self.lost_vals_l[i][j] -= 1
                                else:
                                    # left shift
                                    self.index_offset[i][j] -= 1
                                    if(-self.index_offset[i][j] > 64/2):
                                        self.lost_vals_l[i][j] += 1
                                        quantized_weight[i][j*64 + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                    if(self.lost_vals_r[i][j] > 0):
                                        self.lost_vals_r[i][j] -= 1

                    # print("(2d) total shifts: " + str(shift) + "/" + str(nr))
                    
                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)

                    # print(np.sum(self.lost_vals_r))
                    # print(np.sum(self.lost_vals_l))
                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.block_size, self.error_model)
            if self.an_sim is not None:
                # get tensor sizes
                h = input.shape[2]
                w = input.shape[3]
                kh = self.kernel_size[0]
                kw = self.kernel_size[1] # kernel size
                dh = self.stride[0]
                dw = self.stride[1] # stride
                size = int((h-kh+2*0)/dh+1)

                 # unfold input
                input_b = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride).cuda()
                # unfold kernels
                weight_b = quantized_weight.view(self.out_channels,-1).cuda()

                # nr of neurons
                wm_row = weight_b.shape[0]
                # nr of weights
                wm_col = weight_b.shape[1]
                # nr of columns in image
                im_col = input_b.shape[2]

                if self.performance_mode is not None:
                    # print("performance mode 2d")
                    # print("--")
                    # print("weight shape folded", quantized_weight.shape)
                    # print("--")
                    # print("input shape folded", input.shape)
                    # # print("weight shape", weight_b.shape)
                    # print("input shape unfolded", input_b.shape)
                    # print("h,w", h,w)
                    output_b = torch.zeros(input_b.shape[0], wm_row, im_col).cuda()
                    if self.mapping is not None:
                        # print("mapping in performance mode")
                        # !!! TODO: preliminary hot fix, because ResNet reorganizes output tensors in a weird way
                        # For example:
                        # input shape folded: torch.Size([16, 64, 64, 64])
                        # input shape unfolded: torch.Size([16, 576, 1024])
                        # h,w: 64 64
                        # output shape of output_torch: folded torch.Size([16, 128, 32, 32])
                        # This may not work together well when training with the mapping
                        output_torch = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        # print("cuda_shape", output_b.shape)
                        # print("torch_shape", output_torch.shape)
                        custommac2dmappingdirect.custommac2dmappingdirect(input_b, weight_b, output_b, self.mapping, self.array_size)
                        # output_b = output_b.view(input_b.shape[0], wm_row, h, w)
                        output_b = output_b.view(output_torch.shape)
                        # print("output shape folded", output_torch.shape)
                        output_b = output_b.detach()

                    if self.mapping_distr is not None:
                        # print("mapping in performance mode")
                        output_torch = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        custommac2dmappingdistr.custommac2dmappingdistr(input_b, weight_b, output_b, self.mapping_distr, self.sorted_mapping_idx, self.array_size)
                        # output_b = output_b.view(input_b.shape[0], wm_row, h, w)
                        output_b = output_b.view(output_torch.shape)
                        output_b = output_b.detach()
                    # print("1?")
                    # custommac2dmappingdirect.custommac2dmappingdirect(input_b, weight_b, output_b)
                    # output_b = output_b.view(input_b.shape[0], wm_row, h, w)
                    # output_b = output_b.detach()
                    if self.training is not None:
                        output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        output.data.copy_(output_b.data)
                    else:
                        output = output_b
                    # output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    # size for output buffer
                    buffer_size = int(np.ceil(wm_col/self.array_size))

                    output_b = torch.zeros(input_b.shape[0], wm_row, im_col, buffer_size).cuda()
                    # print("b size", output_b.shape)
                    # print("executing an sim for conv")
                    custommac2d.custommac2d(input_b, weight_b, output_b, self.array_size)
                    # print(output_b)

                    if self.extract_absfreq is not None:
                        # get popcount value (unsigned int, max at self.array_size)
                        # print("1", output_b)
                        output_b_pop = (output_b + self.array_size)/2
                        # print("pop", output_b_pop)
                        output_int = torch.flatten(output_b_pop.int())
                        # print("out int", output_int)
                        self.absfreq += torch.bincount(output_int, minlength=self.array_size+1)
                        # print("absfreq", self.absfreq)
                        # print("pop2", output_b_pop)
                        # transform back to format that is needed by pytorch
                        output_b = 2*output_b_pop - self.array_size
                        # print("2", output_b)

                    if self.mapping is not None:
                        output_b_pop = (output_b + self.array_size)/2
                        # print("pop", output_b_pop)
                        mappingdirect.mappingdirect(output_b_pop, self.mapping)
                        # print("pop2", output_b_pop)
                        # transform back to format that is needed by pytorch
                        output_b = 2*output_b_pop - self.array_size

                    if self.mapping_distr is not None:
                        # print("in mapping_distr")
                        # create buffer for accumulating random values
                        output_b_pop = (output_b + self.array_size)/2
                        mappingdistr.mappingdistr(output_b_pop, self.mapping_distr, self.sorted_mapping_idx)
                        output_b = 2*output_b_pop - self.array_size

                    output_b = torch.sum(output_b, 3)
                    # create the view that PyTorch expects
                    output_torch = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    # output_b = output_b.view(input_b.shape[0], wm_row, h, w)
                    output_b = output_b.view(output_torch.shape)
                    output_b = output_b.detach()

                    # execute standard way, to create computation graph for backprop
                    output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    output.data.copy_(output_b.data)
                    # check correctness
                    # correct = torch.eq(output_b, output)
                    # correct = torch.isclose(output_b, output, atol=1e-3)
                    # correct = (~correct).sum().item()
                    # 0 if tensors match
                    # print("correctness: ", correct)
                    # print("out_b", output_b)
                    # print("out", output)
            else:
                output = F.conv2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            # check quantization case
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            # check whether error model needs to be applied
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
