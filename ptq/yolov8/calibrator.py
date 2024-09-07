import os
import random
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

import config
import util

def get_calib_data_path(data_root, num_samples=1024):
    img_paths = []
    image_list = os.listdir(data_root)
    random.shuffle(image_list)
    cnt = 0
    for img_name in image_list:
        path = data_root + img_name
        img_paths.append(path)
        cnt +=1 
        if cnt == num_samples:
            break
    return img_paths

# For TRT
class TRTCalibDataLoader:
    def __init__(self, data_root, batch_size, calib_count, info):
        self.data_root = data_root
        self.info = info
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.image_list = get_calib_data_path(self.data_root)
        self.calibration_data = np.zeros(
            (self.batch_size, 3, 640, 640), dtype=np.float32
        )

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                image = util.preprocess(image_path, self.info)
                self.calibration_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count


class TRTCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
