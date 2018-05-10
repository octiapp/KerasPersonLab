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
from resnet50 import get_resnet50_base
from bilinear import bilinear_sampler
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
    with KB.tf.variable_scope("repeat"):
        expanded_tensor = KB.tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = KB.tf.tile(expanded_tensor, multiples = multiples)
        repeated_tensor = KB.tf.reshape(tiled_tensor, KB.tf.shape(tensor) * repeats)
    return repeated_tensor


def refine(inputs, num_steps=2):
    base, offsets = inputs
    
    # sample bilinearly
    for _ in range(num_steps):
        base = base + bilinear_sampler(offsets, base)    

    return base


def split_and_refine_mid_offsets(mid_offsets, short_offsets):

    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = KL.Lambda(lambda t: t[:,:,:,2*to_keypoint:2*to_keypoint+2])(short_offsets)
        kp_mid_offsets = KL.Lambda(lambda t: t[:,:,:,2*mid_idx:2*mid_idx+2])(mid_offsets)
        kp_mid_offsets = KL.Lambda(lambda t: refine(t))([kp_mid_offsets, kp_short_offsets])
        output_mid_offsets.append(kp_mid_offsets)

    return KL.Lambda(lambda t: KB.concatenate(t, axis=-1))(output_mid_offsets)


def split_and_refine_long_offsets(long_offsets, short_offsets):

    output_long_offsets = []
    for i in range(config.NUM_KP):
        kp_long_offsets = KL.Lambda(lambda t: t[:,:,:,2*i:2*i+2])(long_offsets)
        kp_short_offsets = KL.Lambda(lambda t: t[:,:,:,2*i:2*i+2])(short_offsets)
        refined_1 = KL.Lambda(lambda t: refine(t))([kp_long_offsets, kp_long_offsets])
        refined_2 = KL.Lambda(lambda t: refine(t))([refined_1, kp_short_offsets])
        output_long_offsets.append(refined_2)

    return KL.Lambda(lambda t: KB.concatenate(t, axis=-1))(output_long_offsets)


def kp_map_loss(x):
    kp_maps_true, kp_maps_pred, unannotated_mask, crowd_mask = x
    loss = KB.mean(KB.binary_crossentropy(kp_maps_true, kp_maps_pred), axis=-1, keepdims=True)
    loss = loss * crowd_mask * unannotated_mask
    return KB.mean(loss, keepdims=True) * config.LOSS_WEIGHTS['heatmap']


def short_offset_loss(x):
    short_offset_true, short_offset_pred, kp_maps_true = x
    loss = KB.abs(short_offset_pred - short_offset_true) / config.KP_RADIUS
    loss = loss * tf_repeat(kp_maps_true, [1,1,1,2])
    loss = KB.sum(loss, keepdims=True) / (KB.sum(kp_maps_true) + KB.epsilon())
    return loss * config.LOSS_WEIGHTS['short']


def mid_offset_loss(x):
    mid_offset_true, mid_offset_pred, kp_maps_true = x
    loss = KB.abs(mid_offset_pred - mid_offset_true) / config.KP_RADIUS
    reordered_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        reordered_maps.extend([kp_maps_true[:,:,:,from_kp], kp_maps_true[:,:,:,from_kp]])
    reordered_maps = KB.stack(reordered_maps, axis=-1)
    loss = loss * reordered_maps
    loss = KB.sum(loss, keepdims=True) / (KB.sum(reordered_maps)+KB.epsilon())
    return loss * config.LOSS_WEIGHTS['mid']


def segmentation_loss(x):
    seg_true, seg_pred, crowd_mask = x
    loss = KB.binary_crossentropy(seg_true, seg_pred)
    loss = loss * crowd_mask
    return KB.mean(loss, keepdims=True) * config.LOSS_WEIGHTS['seg']


def long_offset_loss(x):
    long_offset_true, long_offset_pred, seg_true, crowd_mask, unannotated_mask, overlap_mask = x
    loss = KB.abs(long_offset_pred - long_offset_true) / config.KP_RADIUS
    instances = seg_true * crowd_mask * unannotated_mask * overlap_mask
    loss = loss * instances
    # loss = loss * KB.cast(KB.not_equal(long_offset_true, 0.), KB.floatx())
    # loss = loss * crowd_mask * unannotated_mask * overlap_mask
    loss = KB.sum(loss, keepdims=True) / (KB.sum(instances)+KB.epsilon())
    return loss * config.LOSS_WEIGHTS['long']

def build_personlab_head(features, img_shape, id):

    sfx = '_'+str(id)

    kp_maps = KL.Conv2D(config.NUM_KP, kernel_size=(1,1), activation='sigmoid', name='kp_maps'+sfx)(features)
    short_offsets = KL.Conv2D(2*config.NUM_KP, kernel_size=(1,1), name='short_offsets'+sfx)(features)
    mid_offsets = KL.Conv2D(4*(config.NUM_EDGES), kernel_size=(1,1), name='mid_offsets'+sfx)(features)

    seg_mask = KL.Conv2D(1, kernel_size=(1,1), activation='sigmoid', name='segmentation'+sfx)(features)
    long_offsets = KL.Conv2D(2*config.NUM_KP, kernel_size=(1,1), name='long_offsets'+sfx)(features)

    kp_maps = KL.Lambda(lambda t: KB.tf.image.resize_bilinear(t, img_shape, align_corners=True))(kp_maps)
    short_offsets = KL.Lambda(lambda t: KB.tf.image.resize_bilinear(t, img_shape, align_corners=True))(short_offsets)
    mid_offsets = KL.Lambda(lambda t: KB.tf.image.resize_bilinear(t, img_shape, align_corners=True))(mid_offsets)
    long_offsets = KL.Lambda(lambda t: KB.tf.image.resize_bilinear(t, img_shape, align_corners=True))(long_offsets)
    seg_mask = KL.Lambda(lambda t: KB.tf.image.resize_bilinear(t, img_shape, align_corners=True))(seg_mask)

    mid_offsets = split_and_refine_mid_offsets(mid_offsets, short_offsets)
    long_offsets = split_and_refine_long_offsets(long_offsets, short_offsets)
    
    outputs = [kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask]

    return outputs

def get_losses(ground_truth, outputs):
    kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask = ground_truth
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs

    losses = []
    losses.append(KL.Lambda(kp_map_loss)([kp_maps_true, kp_maps, unannotated_mask, crowd_mask]))
    losses.append(KL.Lambda(short_offset_loss)([short_offset_true, short_offsets, kp_maps_true]))
    losses.append(KL.Lambda(mid_offset_loss)([mid_offset_true, mid_offsets, kp_maps_true]))
    losses.append(KL.Lambda(segmentation_loss)([seg_true, seg_mask, crowd_mask]))
    losses.append(KL.Lambda(long_offset_loss)([long_offset_true, long_offsets, seg_true, crowd_mask, unannotated_mask, overlap_mask]))

    return losses

def get_personlab(train=False, input_tensors=None, with_preprocess_lambda=True, build_base_func=get_resnet101_base,
                  intermediate_supervision=False, intermediate_layer=None, output_stride=config.OUTPUT_STRIDE):
    '''
    Constructs the PersonLab model and returns the model object without compiling

    # Arguments:
        train : (boolean) whether to construct the model for training (`False` if model is only for inference).abs
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
        if input_tensors is None:
            input_img = KL.Input(shape=config.IMAGE_SHAPE)
        else:
            input_img = KL.Input(tensor=input_tensors[0])
    else:
        input_img = KL.Input(shape=(None, None, 3))

    if isinstance(with_preprocess_lambda, KL.Lambda):
        normalized_img = with_preprocess_lambda(input_img)
    else:
        normalized_img = KL.Lambda(lambda t: t/255. - 0.5)(input_img)

    if with_preprocess_lambda not in [False, 0, None]:
        base_model = build_base_func(input_tensor=normalized_img, output_stride=output_stride, return_model=True)
    else:
        base_model = build_base_func(input_tensor=input_img, output_stride=output_stride, return_model=True)
    features = base_model.output

    if not train:
        img_shape = KB.shape(input_img)[1:3]
    else:
        img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]

    if intermediate_supervision:
        if intermediate_layer is None:
            raise ValueError('Intemediate layer name must be supplied if using intermediate supervision!')
        if intermediate_layer not in [l.name for l in base_model.layers]:
            raise ValueError('Layer {} does not exist in the base network'.format(intermediate_layer))
        inter_features = base_model.get_layer(intermediate_layer).output
        inter_outputs = build_personlab_head(inter_features, img_shape, id=1)
    else:
        inter_outputs = []

    outputs = build_personlab_head(features, img_shape, id=intermediate_supervision+1)


    if train:
        mask_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 1)
        
        if input_tensors is None:
            crowd_mask = KL.Input(shape=mask_shape)
            unannotated_mask = KL.Input(shape=mask_shape)
            overlap_mask = KL.Input(shape=mask_shape)
            kp_maps_true = KL.Input(shape=mask_shape[:2]+(config.NUM_KP,))
            short_offset_true = KL.Input(shape=mask_shape[:2]+(2*config.NUM_KP,))
            mid_offset_true = KL.Input(shape=mask_shape[:2]+(4*config.NUM_EDGES,))
            seg_true = KL.Input(shape=mask_shape)
            long_offset_true = KL.Input(shape=mask_shape[:2]+(2*config.NUM_KP,))
        else:
            kp_maps_true = KL.Input(tensor=input_tensors[1])
            short_offset_true = KL.Input(tensor=input_tensors[2])
            mid_offset_true = KL.Input(tensor=input_tensors[3])
            long_offset_true = KL.Input(tensor=input_tensors[4])
            seg_true = KL.Input(tensor=input_tensors[5])
            crowd_mask = KL.Input(tensor=input_tensors[6])
            unannotated_mask = KL.Input(tensor=input_tensors[7])
            overlap_mask = KL.Input(tensor=input_tensors[8])
        
        ground_truth = [kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask]
        losses = get_losses(ground_truth, outputs)

        if intermediate_supervision:
            inter_losses = get_losses(ground_truth, inter_outputs)
        else:
            inter_losses = []
            
        
        model = KM.Model(inputs=[input_img,
                                 kp_maps_true,
                                 short_offset_true,
                                 mid_offset_true,
                                 long_offset_true,
                                 seg_true,
                                 crowd_mask,
                                 unannotated_mask,
                                 overlap_mask],
                         outputs=inter_losses+losses)

        return model

    else:
        return KM.Model(input_img, outputs=inter_outputs+outputs)
