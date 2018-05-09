from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv
import numpy as np
import math
from config import config
from post_proc import get_keypoints

def plot_poses(img, skeletons, save_path=None):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')

    img = img.astype('uint8')
    canvas = img.copy()

    for i in range(17):
        rgba = np.array(cmap(1 - i/17. - 1./34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2]).astype('int32'), 4, colors[i], thickness=-1)

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()

    stickwidth = 4

    for i in range(config.NUM_EDGES):
        for j in range(skeletons):
            edge = config.EDGES[i]
            if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
                continue

            cur_canvas = canvas.copy()
            Y = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            X = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
    plt.imshow(canvas[:,:,:])
    if save_path is not None:
        cv.imwrite(save_path,canvas[:,:,:])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)

def visualize_short_offsets(offsets, keypoint_id, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            keypoint_id = config.KEYPOINTS.index(keypoint_id)
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==keypoint_id]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*keypoint_id:2*keypoint_id+2]
    dists = np.zeros(offsets.shape[:2]+(len(centers),))
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists[:,:,j] = np.sqrt(np.square(idx-c).sum(axis=-1))
    dists = dists.min(axis=-1)
    mask = dists <= radius
    I, J = np.nonzero(mask)

    plt.figure()
    if img is not None:
        plt.imshow(img)
    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.show()


def visualize_mid_offsets(offsets, from_kp, to_kp, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(from_kp, str):
        if not from_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(from_kp))
        else:
            from_kp = config.KEYPOINTS.index(from_kp)
    if isinstance(to_kp, str):
        if not to_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(to_kp))
        else:
            to_kp = config.KEYPOINTS.index(to_kp)

    edge_list = config.EDGES + [edge[::-1] for edge in config.EDGES]
    edge_id = edge_list.index((from_kp, to_kp))
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==from_kp]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*edge_id:2*edge_id+2]
    dists = np.zeros(offsets.shape[:2]+(len(centers),))
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists[:,:,j] = np.sqrt(np.square(idx-c).sum(axis=-1))
    dists = dists.min(axis=-1)
    mask = dists <= radius
    I, J = np.nonzero(mask)

    plt.figure()
    if img is not None:
        plt.imshow(img)
    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.show()

def visualize_long_offsets(offsets, keypoint_id, seg_mask, img=None):
    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            keypoint_id = config.KEYPOINTS.index(keypoint_id)

    kp_offsets = offsets[:,:,2*keypoint_id:2*keypoint_id+2]
    I, J = np.nonzero(seg_mask>0.5)
    
    plt.figure()
    if img is not None:
        plt.imshow(img)
    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.show()