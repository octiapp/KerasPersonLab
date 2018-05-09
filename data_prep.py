import numpy as np
import cv2

from config import config

map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))

def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    if len(keypoints.shape) == 2:
        return keypoints[permute, :]
    return keypoints[:, permute, :]

def get_ground_truth(instance_masks, all_keypoints):
    assert(instance_masks.shape[-1] == len(all_keypoints))

    discs = get_keypoint_discs(all_keypoints)

    kp_maps = make_keypoint_maps(all_keypoints, discs)
    short_offsets = compute_short_offsets(all_keypoints, discs)
    mid_offsets = compute_mid_offsets(all_keypoints, discs)
    long_offsets = compute_long_offsets(all_keypoints, instance_masks)

    return kp_maps, short_offsets, mid_offsets, long_offsets

def get_keypoint_discs(all_keypoints):
    map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    discs = [[] for _ in range(len(all_keypoints))]
    for i in range(config.NUM_KP):
        
        centers = [keypoints[i,:2] for keypoints in all_keypoints if keypoints[i,2] > 0]
        dists = np.zeros(map_shape+(len(centers),))

        for k, center in enumerate(centers):
            dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1))
        if len(centers) > 0:
            inst_id = dists.argmin(axis=-1)
        count = 0
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i,2] > 0:
                discs[j].append(np.logical_and(inst_id==count, dists[:,:,count]<=config.KP_RADIUS))
                count +=1
            else:
                discs[j].append(np.array([]))

    return discs

def make_keypoint_maps(all_keypoints, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    kp_maps = np.zeros(map_shape+(config.NUM_KP,))
    for i in range(config.NUM_KP):
        for j in range(len(discs)):
            if all_keypoints[j][i,2] > 0:
                kp_maps[discs[j][i], i] = 1.
        
    return kp_maps


def compute_short_offsets(all_keypoints, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    r = config.KP_RADIUS
    x = np.tile(np.arange(r, -r-1, -1), [2*r+1, 1])
    y = x.transpose()
    m = np.sqrt(x*x + y*y) <= r
    kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)

    def copy_with_border_check(map, center, disc):
        from_top = max(r-center[1], 0)
        from_left = max(r-center[0], 0)
        from_bottom = max(r-(map_shape[0]-center[1])+1, 0)
        from_right =  max(r-(map_shape[1]-center[0])+1, 0)
        
        cropped_disc = disc[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right]
        map[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right, :][cropped_disc,:] = \
                                    kp_circle[from_top:2*r+1-from_bottom, from_left:2*r+1-from_right, :][cropped_disc,:]

    offsets = np.zeros(map_shape+(2*config.NUM_KP,))
    for i in range(config.NUM_KP):
        # this_offset = np.zeros(shape=map_shape+(2,))
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i,2] > 0:
                copy_with_border_check(offsets[:,:,2*i:2*i+2], (all_keypoints[j][i,0], all_keypoints[j][i,1]), discs[j][i])
        
    return offsets


def compute_mid_offsets(all_keypoints, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    offsets = np.zeros(map_shape+(4*config.NUM_EDGES,))
    for i, edge in enumerate((config.EDGES + [edge[::-1] for edge in config.EDGES])):
        #this_offset = np.zeros(map_shape+(2,))
        for j in range(len(all_keypoints)):
            if all_keypoints[j][edge[0],2] > 0 and all_keypoints[j][edge[1],2] > 0:
                # idx = np.rollaxis(np.indices(map_shape), 0, 3).transpose((1,0,2))
                # dists = np.array([[ all_keypoints[j][edge[1],0], all_keypoints[j][edge[1],1] ]]) - idx
                m = discs[j][edge[0]]
                dists = [[ all_keypoints[j][edge[1],0], all_keypoints[j][edge[1],1] ]] - idx[m,:]
                # this_offset[m,:] = dists[m,:]
        # offsets[:,:,2*i:2*i+2] = this_offset
                offsets[m,2*i:2*i+2] = dists
    
    return offsets


def compute_long_offsets(all_keypoints, instance_masks):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    instance_masks = instance_masks.astype('bool')
    offsets = np.zeros(map_shape+(2*config.NUM_KP,))
    for i in range(config.NUM_KP):
        # this_offset = np.zeros(map_shape+(2,))
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i,2] > 0:
                # idx = np.rollaxis(np.indices(map_shape), 0, 3).transpose((1,0,2))
                m = instance_masks[:,:,j]
                dists = [[ all_keypoints[j][i,0], all_keypoints[j][i,1] ]] - idx[m,:]
                #this_offset[m,:] = dists[m,:]
                offsets[m, 2*i:2*i+2] = dists
        # offsets[:,:,2*i:2*i+2]
    
    overlap = np.sum(instance_masks, axis=-1) >= 2
    offsets[overlap,:] = 0.
    return offsets
