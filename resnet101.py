## This is modified from the version by @flyyufelix here:
# https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294#file-resnet-101_keras-py


from keras import layers as KL
from keras import backend as KB
from keras import models as KM
from keras.optimizers import SGD
from keras import initializers
from keras.engine.topology import get_source_inputs
from keras.engine import Layer, InputSpec
from keras.utils.data_utils import get_file
from frozen_batchnorm import FrozenBatchNorm
from config import config

if config.BATCH_NORM_FROZEN:
    BatchNormalization = FrozenBatchNorm
else:
    BatchNormalization = KL.BatchNormalization

def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=1):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    bn_axis = 3
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=False, dilation_rate=dilation)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)

    x = KL.add([x, input_tensor], name='res' + str(stage) + block)
    x = KL.Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=1):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    bn_axis = 3
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu', name=conv_name_base + '2a_relu')(x)

    # x = KL.ZeroPadding2D((1, 1), name=conv_name_base + '2b_ZeroPadding')(x)
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=False, dilation_rate=dilation)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = KL.add([x, shortcut], name='res' + str(stage) + block)
    x = KL.Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def get_resnet101_base(input_tensor, output_stride=8, return_model=False):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
    bn_axis=3

    input_tensor_source = get_source_inputs(input_tensor)[0]

    # x = KL.ZeroPadding2D((3, 3), name='conv1_ZeroPadding')(input_tensor)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False, padding='same',)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = KL.Activation('relu', name='conv1_relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

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
    for i in range(1,3):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i), dilation=dilation)

    if stride_left > 1:
        strides = (2,2)
        stride_left /= 2
    else:
        strides = (1,1)
        dilation *= 2

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides, dilation=dilation)
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), dilation=dilation)

    if stride_left > 1:
        strides = (2,2)
        stride_left /= 2
    else:
        strides = (1,1)
        dilation *= 2

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides, dilation=dilation)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


    model = KM.Model(input_tensor_source, x)
    weights_path = get_file('resnet101_weights_notop.h5', None, cache_subdir='models')
    model.load_weights(weights_path)

    if return_model:
        return model

    return model.output