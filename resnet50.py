
from keras import layers as KL
from keras import models as KM
from keras import backend as KB
from keras.utils import data_utils
from keras.engine.topology import get_source_inputs
from frozen_batchnorm import FrozenBatchNorm

from config import config

if config.BATCH_NORM_FROZEN:
    BatchNormalization = FrozenBatchNorm
else:
    BatchNormalization = KL.BatchNormalization

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

    x = KL.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2, kernel_size, dilation_rate=dilation,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = KL.add([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
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
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2, kernel_size, dilation_rate=dilation, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = KL.add([x, shortcut])
    x = KL.Activation('relu')(x)
    return x

def get_resnet50_base(input_tensor, output_stride=8, return_model=False):
    if KB.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    input_tensor_source = get_source_inputs(input_tensor)[0]

    x = KL.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)

    current_stride = 4
    stride_left = output_stride / current_stride

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')


    if stride_left > 1:
        strides = (2,2)
        dilation = 1
        stride_left /= 2
    else:
        strides = (1,1)
        dilation = 2

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=strides, dilation=dilation)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', dilation=dilation)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', dilation=dilation)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', dilation=dilation)

    if stride_left > 1:
        strides = (2,2)
        stride_left /= 2
    else:
        strides = (1,1)
        dilation *= 2

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides, dilation=dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation=dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation=dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation=dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation=dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation=dilation)

    if stride_left > 1:
        strides = (2,2)
        stride_left /= 2
    else:
        strides = (1,1)
        dilation *= 2

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides, dilation=dilation)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation=dilation)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation=dilation)

    model = KM.Model(input_tensor_source, x)

    weights_path = data_utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)

    if return_model:
        return model
    return model.output