import os

from model import PersonLab
from tf_data_generator import *
from config import config
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras import backend as KB

from polyak_callback import PolyakMovingAverage
from losses import CombinedLoss

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Only allocate memory on GPUs as needed
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
set_session(tf.Session(config=tf_config))

LOAD_MODEL_FILE = config.LOAD_MODEL_PATH
SAVE_MODEL_FILE = config.SAVE_MODEL_PATH
H5_DATASET = config.H5_DATASET

num_gpus = config.NUM_GPUS
batch_size_per_gpu = config.BATCH_SIZE_PER_GPU
batch_size = num_gpus * batch_size_per_gpu

input_tensor, target_tensor = get_data_input_tensor(batch_size=batch_size)
# for i in range(len(input_tensors)):
input_tensor.set_shape((None,)+config.IMAGE_SHAPE)
target_tensor.set_shape((None,)+config.IMAGE_SHAPE[:2]+(5*config.NUM_KP+4*config.NUM_EDGES+4,))

if num_gpus > 1:
    with tf.device('/cpu:0'):
        model = PersonLab(config, train=True, input_tensor=input_tensor)
else:
    model = PersonLab(config, train=True, input_tensor=input_tensor)

if LOAD_MODEL_FILE is not None:
    model.load_weights(LOAD_MODEL_FILE, by_name=True)

if num_gpus > 1:
    parallel_model = multi_gpu_model(model, num_gpus)
else:
    parallel_model = model

def save_model(epoch, logs):
    model.save_weights(SAVE_MODEL_FILE)


callbacks = [LambdaCallback(on_epoch_end=save_model)]

# if config.POLYAK:
#     def build_save_model():
#         with tf.device('/cpu:0'):
#             save_model = get_personlab(train=True, input_tensors=input_tensors, with_preprocess_lambda=True)
#         return save_model
#     polyak_save_path = '/'.join(config.SAVE_MODEL_FILE.split('/')[:-1]+['polyak_'+config.SAVE_MODEL_FILE.split('/')[-1]])
#     polyak = PolyakMovingAverage(filepath=polyak_save_path, verbose=1, save_weights_only=True,
#                                     build_model_func=build_save_model, parallel_model=True)

#     callbacks.append(polyak)

# The paper uses SGD optimizer with lr=0.0001 together with Polyak-Averaging
# We use the Adam optimizer with the default params instead
combined_loss = CombinedLoss()
parallel_model.compile(target_tensors=[target_tensor], loss=combined_loss.loss_func, optimizer=Adam())

# This allows for real-time print-outs of the individual losses per output
# even though the loss is all combined into one function.
individual_losses = combined_loss.get_separate_losses()
parallel_model.metrics_tensors.extend(individual_losses)
parallel_model.metrics_names.extend(['maps', 'short', 'mid', 'seg', 'long'])

parallel_model.fit(steps_per_epoch=64115//batch_size,
                                epochs=config.NUM_EPOCHS, callbacks=callbacks)
