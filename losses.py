from keras import backend as KB
tf = KB.tf
from config import config


def mean_or_zero(x):
    non_empty = tf.cast(tf.size(x), tf.bool)
    return tf.cond(non_empty, lambda: tf.reduce_mean(x), lambda: 0.0)

def kp_map_loss(x):
    kp_maps_true, kp_maps_pred, unannotated_mask, crowd_mask = x
    inds = tf.where(tf.cast(tf.squeeze(unannotated_mask * crowd_mask, axis=-1), tf.bool)) # inds.shape=(?,3)
    y_true = tf.gather_nd(kp_maps_true, inds) # y_true.shape = (?, 1)
    y_pred = tf.gather_nd(kp_maps_pred, inds)
    loss = KB.binary_crossentropy(y_true, y_pred)

    return mean_or_zero(loss) * config.LOSS_WEIGHTS['heatmap']


def short_offset_loss(x):
    short_offset_true, short_offset_pred, kp_maps_true = x

    inds = tf.where(tf.cast(KB.repeat_elements(kp_maps_true, 2, axis=-1), tf.bool)) # inds.shape=(?,4)
    y_true = tf.gather_nd(short_offset_true, inds)
    y_pred = tf.gather_nd(short_offset_pred, inds)    
    loss = KB.abs(y_pred - y_true) / config.KP_RADIUS

    return mean_or_zero(loss) * config.LOSS_WEIGHTS['short']


def mid_offset_loss(x):
    mid_offset_true, mid_offset_pred, kp_maps_true = x

    # from_kp_inds = [edge[0] for edge in config.EDGES + [edge[::-1] for edge in config.EDGES]]
    # from_kp_inds = [i for i in from_kp_inds for _ in range(2)]
    # inds = tf.where(tf.cast(tf.gather(kp_maps_true, from_kp_inds), tf.bool))
    
    # NOTE: We cannot use the kp_maps to mask here since
    # the kp_maps are "on" even in keypoint location for which the connecting
    # keypoint is not present.
    inds = tf.where(tf.not_equal(mid_offset_true, 0.))

    y_true = tf.gather_nd(mid_offset_true, inds)
    y_pred = tf.gather_nd(mid_offset_pred, inds)
    loss = KB.abs(y_pred - y_true) / config.KP_RADIUS
    
    return mean_or_zero(loss) * config.LOSS_WEIGHTS['mid']


def segmentation_loss(x):
    seg_true, seg_pred, crowd_mask = x

    inds = tf.where(tf.cast(crowd_mask[:,:,:,0], tf.bool))
    y_true = tf.gather_nd(seg_true, inds)
    y_pred = tf.gather_nd(seg_pred, inds)
    loss = KB.binary_crossentropy(y_true, y_pred)
    
    return mean_or_zero(loss) * config.LOSS_WEIGHTS['seg']


def long_offset_loss(x):
    long_offset_true, long_offset_pred, seg_true, crowd_mask, unannotated_mask, overlap_mask = x

    inds = tf.where(tf.cast(seg_true * crowd_mask * unannotated_mask * overlap_mask, tf.bool))
    y_true = tf.gather_nd(long_offset_true, inds)
    y_pred = tf.gather_nd(long_offset_pred, inds)
    loss = KB.abs(y_pred - y_true) / config.KP_RADIUS

    return mean_or_zero(loss) * config.LOSS_WEIGHTS['long']

class CombinedLoss(object):
    def __init__(self, loss_weights=config.LOSS_WEIGHTS):
        self.weights = loss_weights

    def loss_func(self, y_true, y_pred):
        ground_truths = tf.split(y_true, [config.NUM_KP, 2*config.NUM_KP, 4*config.NUM_EDGES,
                                            2*config.NUM_KP, 1, 1, 1, 1], axis=-1)
        preds = tf.split(y_pred, [config.NUM_KP, 2*config.NUM_KP, 4*config.NUM_EDGES,
                                    2*config.NUM_KP, 1], axis=-1)

        kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask = ground_truths
        kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = preds

        self.losses = []
        self.losses.append(kp_map_loss([kp_maps_true, kp_maps, unannotated_mask, crowd_mask]))
        self.losses.append(short_offset_loss([short_offset_true, short_offsets, kp_maps_true]))
        self.losses.append(mid_offset_loss([mid_offset_true, mid_offsets, kp_maps_true]))
        self.losses.append(segmentation_loss([seg_true, seg_mask, crowd_mask]))
        self.losses.append(long_offset_loss([long_offset_true, long_offsets, seg_true, crowd_mask, unannotated_mask, overlap_mask]))
        total_loss = self.losses[0]
        for loss in self.losses[1:]:
            total_loss += loss

        return total_loss

    def get_separate_losses(self):
        return self.losses