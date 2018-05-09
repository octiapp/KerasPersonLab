import numpy as np
from config import config
from data_iterator import RawDataIterator

class DataIteratorBase(object):

    def __init__(self, batch_size = 10, input_shapes=None):

        self.batch_size = batch_size
        if input_shapes is None:
            self.input_shapes = [
                config.IMAGE_SHAPE,
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], config.NUM_KP),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 2*config.NUM_KP),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 4*config.NUM_EDGES),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 2*config.NUM_KP),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
                (config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1], 1),
            ]
        else:
            self.input_shapes = input_shapes

    def gen_raw(self): # this function used for test purposes in py_rmpe_server

        while True:
            yield tuple(self._recv_arrays())

    def gen(self):
        all_batches = [np.zeros(shape=(self.batch_size,)+input_shape) for input_shape in self.input_shapes]

        sample_idx = 0

        for foo in self.gen_raw():

            for k, input in enumerate(foo):
                all_batches[k][sample_idx,...] = input

            sample_idx += 1

            if sample_idx == self.batch_size:
                sample_idx = 0

                yield (all_batches, None)

    def keypoints(self):
        return self.keypoints


class DataIterator(DataIteratorBase):

    def __init__(self, file, shuffle=True, augment=True, batch_size=10, limit=None):

        super(DataIterator, self).__init__(batch_size)

        self.limit = limit
        self.records = 0

        self.raw_data_iterator = RawDataIterator(file, shuffle=shuffle, augment=augment)
        self.generator = self.raw_data_iterator.gen()
        self.num_samples = self.raw_data_iterator.num_keys()


    def _recv_arrays(self):

        while True:

            if self.limit is not None and self.records > self.limit:
                raise StopIteration

            tpl = next(self.generator, None)
            if tpl is not None:
                self.records += 1
                return tpl

            if self.limit is None or self.records < self.limit:
                print("Staring next generator loop cycle")
                self.generator = self.raw_data_iterator.gen()
            else:
                raise StopIteration


