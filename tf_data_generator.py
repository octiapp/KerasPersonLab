import h5py
import numpy as np
import tensorflow as tf
import random

from config import config, TransformationParams
from data_prep import *
from transformer import Transformer, AugmentSelection

# input_shapes = [
#                 config.IMAGE_SHAPE,
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], config.NUM_KP),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 2*config.NUM_KP),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 4*config.NUM_EDGES),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 2*config.NUM_KP),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
#                 (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
#             ]

def read_data(datum, key):

    entry = datum[key]

    assert 'keypoints' in entry.attrs

    kp = entry.attrs['keypoints']
    kp = np.reshape(kp, (-1, config.NUM_KP, 3))
    data = entry.value
    
    img = data[:,:,0:3]
    encoding = data[:,:,3]

    return img, encoding, kp

def transform_data(img, encoding, kp):

    # Decode
    seg_mask = encoding > 0
    encoding = np.unpackbits(np.expand_dims(encoding, axis=-1), axis=-1)
    unannotated_mask = encoding[:,:,0].astype('bool')
    crowd_mask = encoding[:,:,1].astype('bool')
    overlap_mask = encoding[:,:,2].astype('bool')
    encoding[:,:,:3] = 0
    encoding = np.squeeze(np.packbits(encoding, axis=-1))

    num_instances = int(encoding.max())
    instance_masks = np.zeros((encoding.shape+(num_instances,)))
    for i in range(num_instances):
        instance_masks[:,:,i] = encoding==i+1

    ###

    single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]

    aug = AugmentSelection.random()
    num_instances = instance_masks.shape[-1]
    all_masks = np.concatenate([np.stack(single_masks, axis=-1), instance_masks], axis=-1)
    img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug)
    if num_instances > 0:
        instance_masks = all_masks[:,:, -num_instances:]
    seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:,:, :4].transpose((2,0,1))
    unannotated_mask, crowd_mask, overlap_mask = [np.logical_not(m).astype('float32') for m in [unannotated_mask, crowd_mask, overlap_mask]]
    seg_mask, unannotated_mask, crowd_mask, overlap_mask = [np.expand_dims(m, axis=-1) for m in [seg_mask, unannotated_mask, crowd_mask, overlap_mask]]
    if kp.shape[0] > 0:
        kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
    kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)

    input_tensor = img.astype('float32')
    
    output_tensor = np.concatenate([kp_maps.astype('float32'),
                                short_offsets.astype('float32'),
                                mid_offsets.astype('float32'),
                                long_offsets.astype('float32'),
                                seg_mask.astype('float32'),
                                crowd_mask.astype('float32'),
                                unannotated_mask.astype('float32'),
                                overlap_mask.astype('float32')], axis=-1)

    return [input_tensor, output_tensor]


def data_gen():
    h5 = h5py.File(config.H5_DATASET, 'r')
    root = h5['coco']
    keys = root.keys()
    while True:
        random.shuffle(keys)
        for key in keys:
            yield tuple(read_data(root, key))


def get_data_input_tensor(batch_size=16):
    tf_types = 2*(tf.uint8,) + (tf.int32,)          
    with tf.device('/cpu:0'):
        tf_data = tf.data.Dataset.from_generator(data_gen, tf_types)
        tf_data = tf_data.map(lambda img, encoding, kp :
                                tf.py_func(transform_data, [img, encoding, kp], 2*[tf.float32], stateful=False),
                                num_parallel_calls=64)
        tf_data = tf_data.batch(batch_size)
        tf_data = tf_data.prefetch(10)
        tf_iter = tf_data.make_one_shot_iterator()

        batch_tensors = tf_iter.get_next()

    return batch_tensors
