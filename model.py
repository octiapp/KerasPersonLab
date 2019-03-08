"""
Keras PersonLab


(c) Octi Inc
Written by Jacob Richeimer
"""

from keras import layers as KL
from keras import backend as KB
from keras import models as KM
from keras import losses

from resnet101 import get_resnet101_base
from resnet50 import ResNet50
# from bilinear import bilinear_sampler
from config import config
import numpy as np
import tensorflow as tf



def tf_repeat(tensor, repeats):
    """
    From  https://github.com/tensorflow/tensorflow/issues/8246
    
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor


def refine(inputs, num_steps=2):
    base, offsets = inputs
    
    # sample bilinearly
    for _ in range(num_steps):
        base = base + bilinear_sampler(offsets, base)    

    return base

def sync_shapes(tensors):
    shapes = [KB.int_shape(t) for t in tensors]
    if all([s[1] is None for s in shapes]) or all([s[2] is None for s in shapes]):
        return
    heights = [s[1] for s in shapes if s[1]!=None]
    widths = [s[2] for s in shapes if s[2]!=None]
    if len(set(heights))!=1 or len(set(widths))!=1:
        raise ValueError('mismatching spatial dims in sync_shapes')
    target_h, target_w = heights[0], widths[0]
    for i, t in enumerate(tensors):
        if None in shapes[i][1:3]:
            t._keras_shape = shapes[i][:1]+(target_h, target_w)+shapes[i][3:]



def refine_mid_offsets(x):

    # output_mid_offsets = []
    # for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
    #     to_keypoint = edge[1]
    #     kp_short_offsets = KL.Lambda(lambda t: t[:,:,:,2*to_keypoint:2*to_keypoint+2])(short_offsets)
    #     kp_mid_offsets = KL.Lambda(lambda t: t[:,:,:,2*mid_idx:2*mid_idx+2])(mid_offsets)
    #     kp_mid_offsets = KL.Lambda(lambda t: refine(t))([kp_mid_offsets, kp_short_offsets])
    #     output_mid_offsets.append(kp_mid_offsets)

    # return KL.Lambda(lambda t: KB.concatenate(t, axis=-1))(output_mid_offsets)

    mid_offsets, short_offsets = x

    in_shape = tf.shape(short_offsets)
    idx = tf.cast(tf.expand_dims(tf.stack(tf.meshgrid(tf.range(in_shape[1]), tf.range(in_shape[2]), indexing='ij')[::-1], axis=-1), axis=0), tf.float32)
    short_offset_inds = [edge[1] for edge in config.EDGES + [edge[::-1] for edge in config.EDGES]]
    mid_offsets = tf.reshape(mid_offsets, [in_shape[0], in_shape[1], in_shape[2], config.NUM_EDGES*2, 2]) # now shape=(N, H, W, 32, 2)
    mid_offsets = tf.reshape(tf.transpose(mid_offsets, [0,3,1,2,4]), [in_shape[0]*config.NUM_EDGES*2, in_shape[1], in_shape[2], 2])
    mid_offsets += idx
    
    
    short_offsets = tf.transpose(tf.reshape(short_offsets, [in_shape[0], in_shape[1], in_shape[2], config.NUM_KP, 2]), [3,0,1,2,4]) # now shape=(17, batch, H, W, 2)
    short_offsets = tf.gather(short_offsets, short_offset_inds) # now shape=(32, batch, H, W, 2)
    short_offsets = tf.reshape(tf.transpose(short_offsets, [1,0,2,3,4]), [in_shape[0]*config.NUM_EDGES*2, in_shape[1], in_shape[2], 2]) # now shape=(32*batch,H,W,2)

    for _ in range(config.NUM_REFINEMENTS):
        mid_offsets = mid_offsets + tf.contrib.resampler.resampler(short_offsets, mid_offsets)

    mid_offsets -= idx
    mid_offsets = tf.reshape(mid_offsets, [in_shape[0], config.NUM_EDGES*2, in_shape[1], in_shape[2], 2])
    mid_offsets = tf.reshape(tf.transpose(mid_offsets, [0,2,3,1,4]), [in_shape[0], in_shape[1], in_shape[2], 4*config.NUM_EDGES])
    
    return mid_offsets


def refine_long_offsets(x):

    # output_long_offsets = []
    # for i in range(config.NUM_KP):
    #     kp_long_offsets = KL.Lambda(lambda t: t[:,:,:,2*i:2*i+2])(long_offsets)
    #     kp_short_offsets = KL.Lambda(lambda t: t[:,:,:,2*i:2*i+2])(short_offsets)
    #     refined_1 = KL.Lambda(lambda t: refine(t))([kp_long_offsets, kp_long_offsets])
    #     refined_2 = KL.Lambda(lambda t: refine(t))([refined_1, kp_short_offsets])
    #     output_long_offsets.append(refined_2)

    # return KL.Lambda(lambda t: KB.concatenate(t, axis=-1))(output_long_offsets)
    long_offsets, short_offsets = x

    in_shape = tf.shape(long_offsets)
    idx = tf.cast(tf.expand_dims(tf.stack(tf.meshgrid(tf.range(in_shape[1]), tf.range(in_shape[2]), indexing='ij')[::-1], axis=-1), axis=0), tf.float32)
    long_offsets = tf.reshape(long_offsets, [in_shape[0], in_shape[1], in_shape[2], config.NUM_KP, 2])
    long_offsets = tf.reshape(tf.transpose(long_offsets, [0,3,1,2,4]), [in_shape[0]*config.NUM_KP, in_shape[1], in_shape[2], 2])
    long_offsets += idx

    short_offsets = tf.reshape(short_offsets, [in_shape[0], in_shape[1], in_shape[2], config.NUM_KP, 2])
    short_offsets = tf.reshape(tf.transpose(short_offsets, [0,3,1,2,4]), [in_shape[0]*config.NUM_KP, in_shape[1], in_shape[2], 2])

    for _ in range(config.NUM_REFINEMENTS):
        long_offsets = long_offsets + tf.contrib.resampler.resampler(short_offsets, long_offsets)

    long_offsets -= idx
    long_offsets = tf.reshape(long_offsets, [in_shape[0], config.NUM_KP, in_shape[1], in_shape[2], 2])
    long_offsets = tf.reshape(tf.transpose(long_offsets, [0,2,3,1,4]), [in_shape[0], in_shape[1], in_shape[2], 2*config.NUM_KP])

    return long_offsets

def build_personlab_head(features, img_shape, suffix=''):

    if suffix in [None, '']:
        sfx = ''
    else:
        sfx = '_'+str(suffix)

    kp_maps = KL.Conv2D(config.NUM_KP, kernel_size=(1,1), activation='sigmoid', name='kp_maps'+sfx)(features)
    short_offsets = KL.Conv2D(2*config.NUM_KP, kernel_size=(1,1), name='short_offsets'+sfx)(features)
    mid_offsets = KL.Conv2D(4*(config.NUM_EDGES), kernel_size=(1,1), name='mid_offsets'+sfx)(features)

    seg_mask = KL.Conv2D(1, kernel_size=(1,1), activation='sigmoid', name='segmentation'+sfx)(features)
    long_offsets = KL.Conv2D(2*config.NUM_KP, kernel_size=(1,1), name='long_offsets'+sfx)(features)

    kp_maps = KL.Lambda(lambda t: tf.image.resize_bilinear(t, img_shape, align_corners=True))(kp_maps)
    short_offsets = KL.Lambda(lambda t: tf.image.resize_bilinear(t, img_shape, align_corners=True))(short_offsets)
    mid_offsets = KL.Lambda(lambda t: tf.image.resize_bilinear(t, img_shape, align_corners=True))(mid_offsets)
    long_offsets = KL.Lambda(lambda t: tf.image.resize_bilinear(t, img_shape, align_corners=True))(long_offsets)
    seg_mask = KL.Lambda(lambda t: tf.image.resize_bilinear(t, img_shape, align_corners=True))(seg_mask)

    if config.NUM_REFINEMENTS > 0:
        mid_offsets = KL.Lambda(refine_mid_offsets)([mid_offsets, short_offsets])
        long_offsets = KL.Lambda(refine_long_offsets)([long_offsets, short_offsets])

    outputs = [kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask]

    return outputs


def PersonLab(config, train=False, input_tensor=None):
    '''
    Constructs the PersonLab model and returns the model object without compiling

    # Arguments:
        train : (boolean) whether to construct the model for training (`False` if model is only for inference).
            If `True`, the outputs of the network correspond to the losses described in the paper, which need to be
            supplied to the model via Model.add_loss() after this function is called.
            If `True`, the outputs are (1) the keypoint maps, (2) the short-range offsets, (3) the mid-range offsets,
            (4) the long-range offsets, and (5) the binary segmentation mask.
        
        input_tensors: The input tensors to be used. If `None`, new input tensors will be created. If the model
            is constructed in `train` mode, the inputs include all the ground truth as well for a total of nine
            input tensors.

        with_prepocess_lambda: (boolean or Lambda) If `False`, the base network is applied directly on the input provided.
            If `True`, then the input is preprocessed by normalizing to (-0.5, 0.5) range via a Lambda layer.
            An alternative Lambda layer can also be passed as the value for this argument.

        build_base_func: The function that builds the base network. The available options are get_resnet50_base and
            get_resnet101_base 

        intermediate_supervision: (boolean) Whether to employ intermediate supervision by building a PersonLab head
            on top of intermediate features somewhere in the base network. If `True`, then the `intermediate_layer`
            argument must be provided. Additionally, the amount of outputs of the model (or loss outputs if `train=True`)
            will double when using intermediate supervision.

        intermediate_layer: (str) name of intermediate layer, the output of which to use as features for intermediate
            supervision.
    '''
    
    if train:
        if input_tensor is None:
            input_img = KL.Input(input_shape=config.IMAGE_SHAPE)
        else:
            input_img = KL.Input(tensor=input_tensor)
    else:
        input_img = KL.Input(input_shape=(None, None, 3))

    if config.PREPROCESS_LAMBDA:
        normalized_img = KL.Lambda(lambda t: t/255. - 0.5)(input_img)
    else:
        normalized_img = input_img

    
    if config.BACKBONE.lower() not in ['resnet50']:
        raise NotImplementedError()

    if config.BACKBONE.lower() == 'resnet50':
        base_model = ResNet50(input_tensor=normalized_img, output_stride=config.OUTPUT_STRIDE)
    
    features = base_model.output

    if not train:
        img_shape = KB.shape(input_img)[1:3]
    else:
        img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]

    if config.INTERMEDIATE_SUPERVISION and train:
        if config.INTER_LAYER is None:
            raise ValueError('Intemediate layer name must be supplied if using intermediate supervision!')
        if config.INTER_LAYER not in [l.name for l in base_model.layers]:
            raise ValueError('Layer {} does not exist in the base network'.format(intermediate_layer))
        inter_features = base_model.get_layer(intermediate_layer).output
        inter_outputs = build_personlab_head(inter_features, img_shape, suffix='inter')
        inter_outputs = KL.Concatenate(axis=-1)(inter_outputs)

    outputs = build_personlab_head(features, img_shape, suffix=None)
    if train:
        sync_shapes(outputs)
        outputs = KL.Concatenate(axis=-1)(outputs)
        if config.INTERMEDIATE_SUPERVISION:
            outputs = [outputs, inter_outputs]
        
    model = KM.Model(inputs=input_img, outputs=outputs)

    return model