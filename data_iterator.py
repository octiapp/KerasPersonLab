
import h5py
import random
import numpy as np

from config import config, TransformationParams
from data_prep import *
from transformer import Transformer, AugmentSelection

class RawDataIterator:

    def __init__(self, h5file, shuffle = True, augment = True):

        self.h5file = h5file
        self.h5 = h5py.File(self.h5file, "r")
        self.datum = self.h5['coco']
        self.augment = augment
        self.shuffle = shuffle

    def gen(self, dbg=False):

        keys = list(self.datum.keys())

        if self.shuffle:
            random.shuffle(keys)

        for key in keys:
            image, encoding, keypoints = self.read_data(key)

            all_inputs = self.transform_data(image, encoding, keypoints)

            yield all_inputs

    def num_keys(self):
        return len(list(self.datum.keys()))

    def read_data(self, key):

        entry = self.datum[key]

        assert 'keypoints' in entry.attrs

        kp = entry.attrs['keypoints']
        kp = np.reshape(kp, (-1, config.NUM_KP, 3))
        data = entry.value
        
        img = data[:,:,0:3]
        encoding = data[:,:,3]

        return img, encoding, kp

    def transform_data(self, img, encoding, kp):

        # Decode
        seg_mask = encoding > 0
        encoding = np.unpackbits(np.expand_dims(encoding, axis=-1), axis=-1)
        unannotated_mask = encoding[:,:,0].astype('bool')
        crowd_mask = encoding[:,:,1].astype('bool')
        encoding[:,:,:2] = 0
        encoding = np.squeeze(np.packbits(encoding, axis=-1))

        num_instances = int(encoding.max())
        instance_masks = np.zeros((encoding.shape+(num_instances,)))
        for i in range(num_instances):
            instance_masks[:,:,i] = encoding==i

        overlap_mask = np.zeros_like(seg_mask)
        if instance_masks.shape[0] > 1:
            overlap_mask = instance_masks.sum(axis=-1) > 1

    ###
    ###
        single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
        aug = AugmentSelection.random() if self.augment else AugmentSelection.unrandom()
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

        return [img,
                kp_maps,
                short_offsets,
                mid_offsets,
                long_offsets,
                seg_mask,
                crowd_mask,
                unannotated_mask,
                overlap_mask]


    def __del__(self):

        self.h5.close()
