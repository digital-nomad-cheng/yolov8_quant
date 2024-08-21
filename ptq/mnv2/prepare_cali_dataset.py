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
from config import DIR_VAL

# 200 classes in total
IMGS_PER_CLS = 1
TOTAL_CLS_USED = 200

def preprocess(img):
    transforms_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    img = transforms_val(img)
    return img

def get_calib_data_path(data_root):
    img_paths = []
    data_info = pd.read_table(data_root + "val_annotations.txt")
    grouped = data_info.groupby(data_info.columns[1])
    classes = list(grouped.groups.keys())
    for cls in classes[:TOTAL_CLS_USED]:
        group_imgs = grouped.get_group(cls).iloc[:, 0].tolist()
        random.shuffle(group_imgs)
        img_paths += group_imgs[:IMGS_PER_CLS]

    return img_paths

# For Dipoorlet
def get_dipoorlet_calib_dataset():
    data_root = "../tiny-imagenet-200/val/"
    dst_path = "dipoorlet_work_dir/input.1/"

    # Remove previous files in path directory
    files = glob.glob(dst_path+"*")
    for f in files:
        os.remove(f)
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    image_list = get_calib_data_path(data_root)    
    for i, image_path in tqdm(enumerate(image_list), total=len(image_list)):
        image = Image.open(data_root + "images/" + image_path).convert("RGB")
        image = preprocess(image).numpy()
        image.tofile("dipoorlet_work_dir/input.1/" + str(i) + ".bin")


class TRTCalibDataLoader:
    """
    Data loader for tensorrt calibration.
    """
    def __init__(self, batch_size=1, calib_count=1000):
        self.img_root = DIR_VAL + "images/"
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.image_list = get_calib_data_path(DIR_VAL)
        self.calibration_data = np.zeros(
            (self.batch_size, 3, 224, 224), dtype=np.float32
        )

    def reset(self):
        self.index = 0
    
    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                image = Image.open(self.img_root + image_path).convert("RGB")
                image = preprocess(image)
                self.calibration_data[i] = image
                print(image.shape)
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])
    
    def __len__(self):
        return self.calib_count

class TRTCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Generate dataset on cuda for tensorrt calibration.
    """
    def __init__(self, data_loader=None, cache_file="trt.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        if data_loader is None:
            self.data_loader = TRTCalibDataLoader()
            self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
            self.cache_file = cache_file
            self.data_loader.reset()

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

if __name__ == "__main__":
    get_dipoorlet_calib_dataset()
