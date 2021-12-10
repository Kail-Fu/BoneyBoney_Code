import tensorflow as tf
from tensorflow.keras import Sequential
import numpy as np
import math
from aggregation import Aggregation
from subtraction import Subtraction
from subtraction2 import Subtraction2



def conv1x1(out_planes, stride=1):
    ''' convolutions with kernel size as (1,1) '''
    return tf.keras.layers.Conv2D(out_planes, kernel_size=(1,1), strides=stride, padding="same",data_format="channels_first",use_bias=False)



def position(H, W):
    ''' normalizing position coordinates '''
    loc_w = tf.tile(tf.expand_dims(tf.linspace(-1.0, 1.0, W), axis=0),tf.constant([H, 1]))
    loc_h = tf.tile(tf.expand_dims(tf.linspace(-1.0, 1.0, H), axis=1),tf.constant([1, W]))
    loc = tf.expand_dims(tf.concat([tf.expand_dims(loc_w, axis=0), tf.expand_dims(loc_h, axis=0)], 0), axis=0)
    return loc



# # # # --------------------- Training and Testing --------------------- # # # #
class SAM(tf.keras.Model):
    ''' pair-wise self attention model, corresponding to sa_type=0 in orginal model '''
    def __init__(self, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = tf.keras.layers.Conv2D(rel_planes,kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False) #channel-first???? kernel_size = 1
        self.conv2 = tf.keras.layers.Conv2D(rel_planes,kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(out_planes,kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False)

        # pair-wise attention
        self.conv_w = tf.keras.Sequential()
        self.conv_w.add(tf.keras.layers.BatchNormalization()) # rel_planes+2???
        self.conv_w.add(tf.keras.layers.ReLU())
        self.conv_w.add(tf.keras.layers.Conv2D(rel_planes,kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False))
        self.conv_w.add(tf.keras.layers.BatchNormalization()) # rel_planes
        self.conv_w.add(tf.keras.layers.ReLU())
        self.conv_w.add(tf.keras.layers.Conv2D(out_planes // share_planes, kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False))

        self.conv_p = tf.keras.layers.Conv2D(2, kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.softmax = tf.keras.layers.Softmax(axis=-2)

        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def call(self, x):
        # x3 = the right stream of self-attention block - reduce dimension
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        # get the position
        p = self.conv_p(position(x.shape[2], x.shape[3]))
        # w = the left stream of self-attention block - evaluates the attention weights
        w = self.softmax(self.conv_w(tf.concat([self.subtraction2(x1, x2), tf.tile(self.subtraction(p), tf.constant([x.shape[0], 1, 1, 1]))], 1)))
        # aggregate the right and left streams
        x = self.aggregation(x3, w)
        return x



class Bottleneck(tf.keras.Model):
    ''' dimension reduction block that embded SAM model '''
    def __init__(self, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization() # in_planes
        self.sam = SAM(rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = tf.keras.layers.BatchNormalization() # mid_planes
        self.conv = tf.keras.layers.Conv2D(out_planes, kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False)
        self.relu = tf.keras.layers.ReLU()
        self.stride = stride

    def call(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out



class SAN(tf.keras.Model):
    ''' the SAN model that combines '''
    def __init__(self, block, layers, kernels, num_classes):
        super(SAN, self).__init__()

        # layers = number of self-attention blocks (Bottleneck) per layer
        # kernels = footprint size for self-attention = the amount of context gathered by a self-attention operator

        # to store the train loss/accuracy per batch and test loss/accuracy per epoch
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.test_loss_list = []
        self.test_accuracy_list = []

        c = 64 # num of result channels
        self.conv_in, self.bn_in = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.conv0, self.bn0 = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.layer0 = self._make_layer(block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.layer1 = self._make_layer(block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.layer2 = self._make_layer(block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.layer3 = self._make_layer(block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c), tf.keras.layers.BatchNormalization() # c
        self.layer4 = self._make_layer(block, c, layers[4], kernels[4])

        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, data_format="channels_first")

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")
        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def _make_layer(self, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return tf.keras.Sequential(layers)

    def call(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))

        # each state: multiple self-attention blocks <= self.layerx
        # bridged by transition layers - expand channel dim; reduce spatial resolution <= self.pool + self.relu
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        # classification layer
        x = self.avgpool(x)
        x = tf.reshape(x,[x.shape[0], -1])
        x = self.fc(x)
        return x


    """ tried to implement loss and accuracy functions inside the SAN model;
    in the final implementation, replaced by tf.keras.metrics.mean and tf.keras.metrics.CategoricalAccuracy"""
    # ## from hw2,
    # def loss(self, probs, labels):
    #     return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
    
    # def accuracy(self, probs, labels):
    #     correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
    #     return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    #     # return accuracy_score(y_true=labels, y_pred=logits)



def san(layers, kernels, num_classes):
    ''' a convient way to call SAN model, with self attention blocks set to Bottleneck'''
    model = SAN(Bottleneck, layers, kernels, num_classes)
    return model
