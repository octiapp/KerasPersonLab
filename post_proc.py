import numpy as np
from config import config
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter

def iterative_bfs(graph, start, path=[]):
    '''iterative breadth first search from start'''
    q=[(None,start)]
    visited = []
    while q:
        v=q.pop(0)
        if not v[1] in visited:
            visited.append(v[1])
            path=path+[v]
            q=q+[(v[1], w) for w in graph[v[1]]]
    return path

def accumulate_votes(votes, shape):
    xs = votes[:,0]
    ys = votes[:,1]
    ps = votes[:,2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps*(1.-dx)*(1.-dy)
    tr_vals = ps*dx*(1.-dy)
    bl_vals = ps*dy*(1.-dx)
    br_vals = ps*dy*dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    heatmap = np.asarray(coo_matrix( (data[good_inds], (I[good_inds],J[good_inds])), shape=shape ).todense())
    return heatmap


def compute_heatmaps(kp_maps, short_offsets):
    heatmaps = []
    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))
    for i in range(config.NUM_KP):
        this_kp_map = kp_maps[:,:,i:i+1]
        votes = idx + short_offsets[:,:,2*i:2*i+2]
        votes = np.reshape(np.concatenate([votes, this_kp_map], axis=-1), (-1, 3))
        heatmaps.append(accumulate_votes(votes, shape=map_shape) / (np.pi*config.KP_RADIUS**2))

    return np.stack(heatmaps, axis=-1)

def get_keypoints(heatmaps):
    keypoints = []
    for i in range(config.NUM_KP):
        peaks = maximum_filter(heatmaps[:,:,i], footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmaps[:,:,i]
        peaks = zip(*np.nonzero(peaks))
        keypoints.extend([{'id': i, 'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1], i]} for peak in peaks])
        keypoints = [kp for kp in keypoints if kp['conf'] > config.PEAK_THRESH]
    return keypoints

## THIS IS THE ALGORITHM DESCRIBED IN THE PAPER:

# def group_skeletons(keypoints, mid_offsets, heatmaps):
#     keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
#     skeletons = []
#     dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

#     skeleton_graph = {i:[] for i in range(config.NUM_KP)}
#     for i in range(config.NUM_KP):
#         for j in range(config.NUM_KP):
#             if (i,j) in config.EDGES or (j,i) in config.EDGES:
#                 skeleton_graph[i].append(j)
#                 skeleton_graph[j].append(i)
    
#     for kp in keypoints:
#         if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 4 for s in skeletons]):
#             continue
#         this_skel = np.zeros((config.NUM_KP, 3))
#         this_skel[kp['id'], :2] = kp['xy']
#         this_skel[kp['id'], 2] = heatmaps[int(kp['xy'][1]), int(kp['xy'][0]), kp['id']]
#         path = iterative_bfs(skeleton_graph, kp['id'])[1:]
#         for edge in path:
#             if this_skel[edge[0],2] == 0:
#                 continue
#             mid_idx = dir_edges.index(edge)
#             offsets = mid_offsets[:,:,2*mid_idx:2*mid_idx+2]
#             from_kp = tuple(this_skel[edge[0],:2].astype('int32'))
#             this_skel[edge[1],:2] = this_skel[edge[0],:2] + offsets[from_kp[1], from_kp[0], :]
#             this_skel[edge[1], 2] = heatmaps[int(this_skel[edge[1],1]), int(this_skel[edge[1],0]), edge[1]]

#         skeletons.append(this_skel)

#     return skeletons

def group_skeletons(keypoints, mid_offsets):
    keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
    skeletons = []
    dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

    skeleton_graph = {i:[] for i in range(config.NUM_KP)}
    for i in range(config.NUM_KP):
        for j in range(config.NUM_KP):
            if (i,j) in config.EDGES or (j,i) in config.EDGES:
                skeleton_graph[i].append(j)
                skeleton_graph[j].append(i)
    
    while len(keypoints) > 0:
        kp = keypoints.pop(0)
        if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 10 for s in skeletons]):
            continue
        this_skel = np.zeros((config.NUM_KP, 3))
        this_skel[kp['id'], :2] = kp['xy']
        this_skel[kp['id'], 2] = kp['conf']
        path = iterative_bfs(skeleton_graph, kp['id'])[1:]
        for edge in path:
            if this_skel[edge[0],2] == 0:
                continue
            mid_idx = dir_edges.index(edge)
            offsets = mid_offsets[:,:,2*mid_idx:2*mid_idx+2]
            from_kp = tuple(np.round(this_skel[edge[0],:2]).astype('int32'))
            proposal = this_skel[edge[0],:2] + offsets[from_kp[1], from_kp[0], :]
            matches = [(i, keypoints[i]) for i in range(len(keypoints)) if keypoints[i]['id'] == edge[1]]
            matches = [match for match in matches if np.linalg.norm(proposal-match[1]['xy']) <= 32]
            if len(matches) == 0:
                continue
            matches.sort(key=lambda m: np.linalg.norm(m[1]['xy']-proposal))
            to_kp = np.round(matches[0][1]['xy']).astype('int32')
            to_kp_conf = matches[0][1]['conf']
            keypoints.pop(matches[0][0])
            this_skel[edge[1],:2] = to_kp
            this_skel[edge[1], 2] = to_kp_conf

        skeletons.append(this_skel)

    return skeletons


def get_instance_masks(skeletons, seg_mask, long_offsets, threshold=True):
    map_shape = seg_mask.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))
    features = np.tile(idx, config.NUM_KP) + long_offsets
    num_skels = len(skeletons)

    p_i, p_j = np.nonzero(seg_mask>0.5)
    n = len(p_i)
    probs = np.zeros((n, num_skels))

    for j in range(num_skels):
        scale = (skeletons[j].max(axis=0) - skeletons[j].min(axis=0))[:2].prod()
        scale = np.sqrt(scale)
        this_prob = np.zeros((n,))
        norm_factor = 0.
        for k in range(config.NUM_KP):
            if skeletons[j][k,2] == 0:
                continue
            dists = features[p_i,p_j,2*k:2*k+2] - np.array([[ skeletons[j][k,0], skeletons[j][k,1] ]])
            p = np.sqrt(np.square(dists).sum(axis=-1))
            p *= skeletons[j][k,2] / scale
            this_prob += p
            norm_factor += skeletons[j][k,2]
        probs[:, j] = this_prob / norm_factor
        
    P = 1000.*np.ones(map_shape+(num_skels,))
    P[p_i, p_j, :] = probs
    
    masks = np.zeros(map_shape+(num_skels,))
    masks[idx[:,:,1].flatten(), idx[:,:,0].flatten(), P.argmin(axis=-1).flatten()] = 1
    # This should really use the instance mask distance threshold,
    # but for now is just assigns all pixels in the seg mask to an instance
    # mask for which the metric distance is lowest
    if threshold:
        masks[P.min(axis=-1) > config.INSTANCE_SEG_THRESH, :] = 0
    else:
        masks[P.min(axis=-1) > 999., :] = 0
    return [np.squeeze(m) for m in np.split(masks, num_skels, axis=-1)]


def get_skeletons_and_masks(outputs):
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    heatmaps = compute_heatmaps(kp_maps, short_offsets)
    for i in range(config.NUM_KP):
        heatmaps[:,:,i] = gaussian_filter(heatmaps[:,:,i], sigma=2)
    pred_kp = get_keypoints(heatmaps)
    skeletons = group_skeletons(pred_kp, mid_offsets, kp_maps)
    instance_masks = get_instance_masks(skeletons, seg_mask, long_offsets)

    return skeletons, instance_masks