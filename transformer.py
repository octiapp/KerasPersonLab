import numpy as np
from math import cos, sin, pi
import cv2
import random

from config import config, TransformationParams
from data_prep import map_coco_to_personlab

class AugmentSelection:

    def __init__(self, flip=False, degree = 0., crop = (0,0), scale = 1.):
        self.flip = flip
        self.degree = degree #rotate
        self.crop = crop #shift actually
        self.scale = scale

    @staticmethod
    def random():
        flip = random.uniform(0.,1.) > TransformationParams.flip_prob
        degree = random.uniform(-1.,1.) * TransformationParams.max_rotate_degree
        scale = (TransformationParams.scale_max - TransformationParams.scale_min)*random.uniform(0.,1.)+TransformationParams.scale_min
        x_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);
        y_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    def affine(self, center=(config.IMAGE_SHAPE[1]//2, config.IMAGE_SHAPE[0]//2) , scale_self=1.):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        A = self.scale * cos(self.degree / 180. * pi )
        B = self.scale * sin(self.degree / 180. * pi )

        # scale_size = TransformationParams.target_dist / scale_self * self.scale
        # scale_size = TransformationParams.target_dist / self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        # scale = np.array( [[ scale_size, 0, 0 ],
        #                    [ 0, scale_size, 0 ],
        #                    [  0, 0, 1. ] ])

        flip = np.array( [[ -1 if self.flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., config.IMAGE_SHAPE[1]//2],
                                   [ 0., 1., config.IMAGE_SHAPE[0]//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(rotate).dot(center2zero)

        return combined[0:2]

class Transformer:

    @staticmethod
    def transform(img, masks, keypoints, aug=AugmentSelection.random()):

        # warp picture and mask
        M = aug.affine(center=(img.shape[1]//2, img.shape[0]//2))
        cv_shape = (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])

        # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
        # print(img.shape)
        # for i, img in enumerate(input_transform_targets):
        img = cv2.warpAffine(img, M, cv_shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        
        # concat = np.stack(output_transform_targets, axis=-1)
        masks = cv2.warpAffine(masks, M, cv_shape, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # warp key points
        #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
        #update: may be we don't need it actually, original code removed part sliced more than half totally, may be we should keep it
        keypoints = map_coco_to_personlab(keypoints)
        original_points = keypoints.copy()
        # print keypoints
        original_points[:,:,2]=1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
        keypoints[:,:,0:2]=converted_points

        cropped_kp = keypoints[:,:,0] >= config.IMAGE_SHAPE[1]
        cropped_kp = np.logical_or(cropped_kp, keypoints[:,:,1] >= config.IMAGE_SHAPE[0])
        cropped_kp = np.logical_or(cropped_kp, keypoints[:,:,0] < 0)
        cropped_kp = np.logical_or(cropped_kp, keypoints[:,:,1] < 0)

        keypoints[cropped_kp, 2] = 0
        

        # we just made image flip, i.e. right leg just became left leg, and vice versa

        if aug.flip:
            tmpLeft = keypoints[:, config.LEFT_KP, :]
            tmpRight = keypoints[:, config.RIGHT_KP, :]
            keypoints[:, config.LEFT_KP, :] = tmpRight
            keypoints[:, config.RIGHT_KP, :] = tmpLeft

        # print keypoints
        return img, masks, keypoints

