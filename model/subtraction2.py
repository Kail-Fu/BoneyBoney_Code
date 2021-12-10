import tensorflow as tf
from unfold import tensorflow_unfold


class Subtraction2(tf.keras.Model):
    ''' get the relative position between two feature vectors
        adapted based on test_subtraction_zeropad function in lib/sa/functions/subtraction_zeropad.py of the original code '''
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input1, input2):
        n, c, in_height, in_width = input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3]
        padding = (self.dilation * (self.kernel_size - 1) + 1) // 2
        out_height = int((in_height + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        out_width = int((in_width + 2 * padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)

        unfold_i = tensorflow_unfold(input1, 1, self.dilation, 0, self.stride)
        unfold_j = tensorflow_unfold(input2, self.kernel_size, self.dilation, padding, self.stride)
        y2 = tf.reshape(unfold_i,[n, c, 1, out_height * out_width]) - tf.reshape(unfold_j,[n, c, pow(self.kernel_size, 2), out_height * out_width])
        
        return y2