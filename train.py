import os

from model import get_personlab
from tf_data_generator import *
from config import config
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras import backend as KB

from polyak_callback import PolyakMovingAverage

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Only allocate memory on GPUs as used
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
set_session(tf.Session(config=tf_config))

LOAD_MODEL_FILE = config.LOAD_MODEL_PATH
SAVE_MODEL_FILE = config.SAVE_MODEL_PATH
H5_DATASET = config.H5_DATASET

num_gpus = config.NUM_GPUS
batch_size_per_gpu = config.BATCH_SIZE_PER_GPU
batch_size = num_gpus * batch_size_per_gpu

input_tensors = get_data_input_tensor(batch_size=batch_size)
for i in range(len(input_tensors)):
    input_tensors[i].set_shape((None,)+input_shapes[i])

if num_gpus > 1:
    with tf.device('/cpu:0'):
        model = get_personlab(train=True, input_tensors=input_tensors, with_preprocess_lambda=True)
else:
    model = get_personlab(train=True, input_tensors=input_tensors, with_preprocess_lambda=True)

if LOAD_MODEL_FILE is not None:
    model.load_weights(LOAD_MODEL_FILE, by_name=True)

if num_gpus > 1:
    parallel_model = multi_gpu_model(model, num_gpus)
else:
    parallel_model = model

for loss in parallel_model.outputs:
    parallel_model.add_loss(loss)

def save_model(epoch, logs):
    model.save_weights(SAVE_MODEL_FILE)

# In Keras, this metric will not be computed for this model, since the outputs have no targets.
# Only by commenting out that restriction in the Keras code will allow the display of these metrics
# which can be used to monitor the individual losses.
def identity_metric(y_true, y_pred):
    return KB.mean(y_pred)


callbacks = [LambdaCallback(on_epoch_end=save_model)]

if config.POLYAK:
    def build_save_model():
        with tf.device('/cpu:0'):
            save_model = get_personlab(train=True, input_tensors=input_tensors, with_preprocess_lambda=True)
        return save_model
    polyak_save_path = '/'.join(config.SAVE_MODEL_FILE.split('/')[:-1]+['polyak_'+config.SAVE_MODEL_FILE.split('/')[-1]])
    polyak = PolyakMovingAverage(filepath=polyak_save_path, verbose=1, save_weights_only=True,
                                    build_model_func=build_save_model, parallel_model=True)

    callbacks.append(polyak)

# The paper uses SGD optimizer with lr=0.0001
parallel_model.compile(target_tensors=None, loss=None, optimizer=Adam(), metrics=[identity_metric])
parallel_model.fit(steps_per_epoch=64115//batch_size,
                                epochs=config.NUM_EPOCHS, callbacks=callbacks)
