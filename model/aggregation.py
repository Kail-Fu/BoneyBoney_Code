import tensorflow as tf
from unfold import tensorflow_unfold

class Aggregation(tf.keras.Model):
    ''' aggregate the left and right streams of slef-attention block by Hadamard product
        adapted based on test_aggregation_zeropad function in lib/sa/functions/aggregation_zeropad.py of the original code'''
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input, weight): 
        # Inspired by test_aggregation_zeropad because it's functionally same as aggregation_zeropad 
        # ("same_padding" default in tensorflow is zeropad)
        n, c_x, c_w, in_height, in_width = input.shape[0], input.shape[1], weight.shape[1], input.shape[2], input.shape[3]
        padding = (self.dilation * (self.kernel_size - 1) + 1) // 2
        out_height = int((in_height + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        out_width = int((in_width + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)

        unfold_j = tensorflow_unfold(input, self.kernel_size, self.dilation, padding, self.stride)
        x2 = tf.reshape(unfold_j, [n, c_x // c_w, c_w, pow(self.kernel_size, 2), out_height * out_width])
        y2 = tf.reshape(tf.math.reduce_sum((tf.expand_dims(weight,axis=1) * x2),axis=-2),[n, c_x, out_height, out_width])
        return y2