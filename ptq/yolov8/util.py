import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def preprocess(img_path, info):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    info.update({"img_height": img_height, "img_width": img_width})
    img = letter_box(img, (info["input_width"], info["input_height"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img   

def letter_box(img, new_shape):
    """LetterBox transform image to new_shape while 
    keeping the aspect ratio.
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img

def print_coco_map(anno_json, pred_json):
    anno = COCO(anno_json)
    pred = COCO(pred_json)
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

