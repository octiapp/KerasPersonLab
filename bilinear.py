
## Edited version originally from here:
# https://github.com/iwyoo/tf-bilinear_sampler/blob/master/bilinear_sampler.py

import tensorflow as tf

def bilinear_sampler(x, v):

  def _get_grid_array(N, H, W, h, w):
    N_i = tf.range(N)
    H_i = tf.range(h+1, h+H+1)
    W_i = tf.range(w+1, w+W+1)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]

    return n, h, w

  shape = tf.shape(x) # TRY : Dynamic shape
  N = shape[0]
  H_ = H = shape[1]
  W_ = W = shape[2]
  h = w = 0

  
  x = tf.pad(x,
    ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
  
  vx, vy = tf.split(v, 2, axis=3)
  

  n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]

  vx0 = tf.floor(vx)
  vy0 = tf.floor(vy)
  vx1 = tf.ceil(vx)
  vy1 = tf.ceil(vy) # [N, H, W, 1]

  iy0 = vy0 + h
  iy1 = vy1 + h
  ix0 = vx0 + w
  ix1 = vx1 + w

  H_f = tf.cast(H_, tf.float32)
  W_f = tf.cast(W_, tf.float32)
  mask = tf.less(ix0, 1)
  mask = tf.logical_or(mask, tf.less(iy0, 1))
  mask = tf.logical_or(mask, tf.greater(ix1, W_f))
  mask = tf.logical_or(mask, tf.greater(iy1, H_f))

  iy0 = tf.where(mask, tf.zeros_like(iy0), iy0)
  iy1 = tf.where(mask, tf.zeros_like(iy1), iy1)
  ix0 = tf.where(mask, tf.zeros_like(ix0), ix0)
  ix1 = tf.where(mask, tf.zeros_like(ix1), ix1)


  i00 = tf.concat([n, iy0, ix0], 3)
  i01 = tf.concat([n, iy1, ix0], 3)
  i10 = tf.concat([n, iy0, ix1], 3)
  i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
  i00 = tf.cast(i00, tf.int32)
  i01 = tf.cast(i01, tf.int32)
  i10 = tf.cast(i10, tf.int32)
  i11 = tf.cast(i11, tf.int32)

  x00 = tf.gather_nd(x, i00)
  x01 = tf.gather_nd(x, i01)
  x10 = tf.gather_nd(x, i10)
  x11 = tf.gather_nd(x, i11)

  dx = tf.cast(vx - vx0, tf.float32)
  dy = tf.cast(vy - vy0, tf.float32)
  
  w00 = (1.-dx) * (1.-dy)
  w01 = (1.-dx) * dy
  w10 = dx * (1.-dy)
  w11 = dx * dy
  
  output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

  return output