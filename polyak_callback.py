## modified from https://github.com/alno/kaggle-allstate-claims-severity/blob/master/keras_util.py

# This is not an ideal implementation of Polyak averaging.
# It adds a significant wait time to the end of each epoch when it saves
# a copy of the latest moving average version of the model.

import numpy as np
import scipy.sparse as sp

from keras import backend as K
from keras import models as KM
from keras.callbacks import Callback
from keras.models import load_model

import sys
import warnings


class PolyakMovingAverage(Callback):
    
    def __init__(self, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={},
                 build_model_func=None, parallel_model=True):
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        self.parallel_model = parallel_model
        self.build_model_func = build_model_func
        super(PolyakMovingAverage, self).__init__()

        self.iter_count = 0L

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf


    def on_train_begin(self, logs={}):
        self.inner_model = None
        if self.parallel_model:

            for l in self.model.layers:
                if isinstance(l, KM.Model):
                    self.inner_model = l
            if self.inner_model is None:
                raise ValueError('No Inner model found in parallel model passed to polyak callback')
        else:
            self.inner_model = self.model

        self.sym_trainable_weights = self.inner_model.trainable_weights
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        self.iter_count += 1L
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0/self.iter_count) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        model2 = self._make_mv_model(filepath)
                        if self.save_weights_only:
                            model2.save_weights(filepath, overwrite=True)
                        else:
                            model2.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))
                model2 = self._make_mv_model(filepath)
                if self.save_weights_only:
                    model2.save_weights(filepath, overwrite=True)
                else:
                    model2.save(filepath, overwrite=True)

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        if build_model_func is None:
            self.inner_model.save(filepath, overwrite=True)
            model2 = load_model(filepath, custom_objects=self.custom_objects)
        else:
            self.inner_model.save_weights(filepath, overwrite=True)
            model2 = self.build_model_func()
            model2.load_weights(filepath)

        for w2, w in zip(model2.trainable_weights, self.inner_model.trainable_weights):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])

        return model2