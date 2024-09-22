import cv2
import glob
import numpy as np

files = glob.glob("/media/vincent/FAFC59F8FC59B01D/datasets/coco_minitrain_25k/images/train2017/*.jpg")[:10]
img_datas = []
for idx, file in enumerate(files):
    bgr_img = cv2.imread(file)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, dsize=(640,640))
    extend_batch_size_img = resized_img[np.newaxis, :]
    normalized_img = extend_batch_size_img / 255.0 
    print(
        f'{str(idx+1).zfill(2)}. extend_batch_size_img.shape: {extend_batch_size_img.shape}'
    )
    img_datas.append(normalized_img.astype(np.float32))
calib_datas = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_datas.shape}')
np.save(file='data/calibdata.npy', arr=calib_datas)

loaded_data = np.load('data/calibdata.npy')
print(f'loaded_data.shape: {loaded_data.shape}')