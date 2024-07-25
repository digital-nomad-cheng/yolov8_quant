import os
import glob
import random
from tqdm.auto import tqdm

from PIL import Image
import pandas as pd
from torchvision import transforms

# 200 classes in total
IMGS_PER_CLS = 10
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

if __name__ == "__main__":
    get_dipoorlet_calib_dataset()
