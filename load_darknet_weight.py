import numpy as np
import torch
import torch.nn as nn

from model.darknet import YoloV3, Darknet53, ConvBlock

def copy_weights(bn, conv, ptr, weights, use_bn=True):
    if use_bn:
        num_bn_biases = bn.bias.numel()
        
        #Load the weights
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases
        
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases
        
        #Cast the loaded weights into dims of model weights. 
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        #Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)
    else:
        #Number of biases
        num_biases = conv.bias.numel()
    
        #Load the weights
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        
        #reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)
        
        #Finally copy the data
        conv.bias.data.copy_(conv_biases)
    
    #Let us load the weights for the Convolutional layers
    num_weights = conv.weight.numel()
    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
    ptr = ptr + num_weights

    conv_weights = conv_weights.view_as(conv.weight.data)
    conv.weight.data.copy_(conv_weights)
    return ptr


def load_weights(darknet, file):

    fp = open(file, "rb")
    header = np.fromfile(fp, dtype = np.int32, count=5)
    weights = np.fromfile(fp, dtype = np.float32)
    ptr = 0

    
    first_conv = darknet.conv1
    bn = first_conv.norm
    conv = first_conv.conv

    ptr = copy_weights(bn, conv, ptr, weights)


    blocks = [ 
        darknet.conv1, darknet.conv2, darknet.residual_block1,
        darknet.conv3, darknet.residual_block2,  darknet.conv4,
        darknet.residual_block3, darknet.conv5, darknet.residual_block4,
        darknet.conv6, darknet.residual_block5
    ]
    for layer in blocks:
        
        if isinstance(layer, ConvBlock):
            ptr = copy_weights(layer.norm, layer.conv, ptr, weights)
            print(f" is a Convolutional Block")

        elif isinstance(layer, nn.Sequential):
            print(f" is a Residual Block")
            for residual in layer:
                ptr = copy_weights(residual.conv_batch1.norm, residual.conv_batch1.conv, ptr, weights)
                ptr = copy_weights(residual.conv_batch2.norm, residual.conv_batch2.conv, ptr, weights)

    print(" weight init")



if __name__ == '__main__':
    weight_file = "/home/anandhu/Documents/works/yoloV3/weights/yolov3.weights"
    darknet53 = Darknet53()

    load_weights(darknet53, weight_file)

    torch.save(darknet53.state_dict(), "darknet53.pth")