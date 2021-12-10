import tensorflow as tf
import math


def tensorflow_unfold(input, kernal_size, dilation, padding, stride):
    '''tensorflow version of pytorch unfold'''
    # Implement from formula provided in https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # power of 2: (from pytorch doc) If kernel_size, dilation, padding or stride is an int or a tuple of length 1, their values will be replicated across all spatial dimensions.
    output_dim_2 = input.shape[1]*pow(kernal_size,2) # Output: (N, C \times \prod(\text{kernel\_size}), L)(N,C×∏(kernel_size),L)
    spatial_size = input.shape[2]
    denominator = spatial_size + 2 * padding - dilation * (kernal_size - 1) - 1 # -1 / +1
    l = math.floor(pow(denominator/stride + 1 , 2) )
    return tf.image.resize(input, [output_dim_2, int(l)])[:,:,:,0]
