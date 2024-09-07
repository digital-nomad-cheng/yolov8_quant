import os
import glob
import random
from tqdm.auto import tqdm

from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import config
import util

# 200 classes in total
IMGS_PER_CLS = 1
TOTAL_CLS_USED = 200


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

# For Dipoorlet
def get_dipoorlet_calib_dataset():
    data_root = config.coco_val_path
    dst_path = config.dipoorlet_cali_dataset_path

    # Remove previous files in path directory
    files = glob.glob(dst_path+"*")
    for f in files:
        os.remove(f)
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    image_list = get_calib_data_path(data_root)    
    for i, image_path in tqdm(enumerate(image_list), total=len(image_list)):
        image = util.preprocess(image_path, config.info)
        image.tofile(dst_path + str(i) + ".bin")


if __name__ == "__main__":
    get_dipoorlet_calib_dataset()
