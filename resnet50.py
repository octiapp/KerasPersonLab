# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from keras import layers as KL
from keras import backend as KB
from keras import models as KM
from keras.optimizers import SGD
from keras import initializers
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs
from keras.engine import Layer, InputSpec
from keras.utils.data_utils import get_file

from numpy import log2


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=1):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if KB.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filters1, (1, 1), dilation_rate=dilation, name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2, kernel_size, dilation_rate=dilation,
               padding='same', name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), dilation_rate=dilation, name=conv_name_base + '2c')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = KL.add([x, input_tensor])
    x = KL.Activation('relu', name='out_relu_stage_'+str(stage)+'_block_'+block)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if KB.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation,
               name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = KL.add([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


def ResNet50(output_stride=32, weights='imagenet',
             input_tensor=None, input_shape=(None, None, 3)):

    strides = 3 * [2]
    dilations = 3 * [1]
    assert output_stride >= 4 
    unstride = 5 - int(log2(output_stride))
    if unstride > 0:
        strides[-unstride:] = unstride*[1]
        dil = 2**unstride
        for i in range(unstride):
            dilations[-(i+1)] = dil
            dil = dil // 2

    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not KB.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    if KB.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = KL.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = KL.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = KL.Activation('relu', name='relu_x2')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=strides[0])
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', dilation=dilations[0])
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', dilation=dilations[0])
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', dilation=dilations[0])

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides[1], dilation=dilations[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation=dilations[1])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation=dilations[1])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation=dilations[1])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation=dilations[1])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation=dilations[1])

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides[2], dilation=dilations[1])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation=dilations[2])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation=dilations[2])


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = KM.Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)


    return model
