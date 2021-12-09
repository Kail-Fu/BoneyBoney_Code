# from torch import nn
# from torch.nn.modules.utils import _pair

# from .. import functional as F
import tensorflow as tf
import san

class Aggregation(tf.keras.Model):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input, weight): # Inspired by test_aggregation_zeropad because it's functionally same as aggregation_zeropad
        # return F.aggregation(input, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
        n, c_x, c_w, in_height, in_width = input.shape[0], input.shape[1], weight.shape[1], input.shape[2], input.shape[3]
        padding = (self.dilation * (self.kernel_size - 1) + 1) // 2
        out_height = int((in_height + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        out_width = int((in_width + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)

        unfold_j = san.tensorflow_unfold(input, self.kernel_size, self.dilation, padding, self.stride)
        x2 = tf.reshape(unfold_j, [n, c_x // c_w, c_w, pow(self.kernel_size, 2), out_height * out_width])
        y2 = tf.reshape(tf.math.reduce_sum((tf.expand_dims(weight,axis=1) * x2),axis=-2),[n, c_x, out_height, out_width])
        return y2