from loguru import logger
from pycocotools.coco import COCO
import os
import json
import time
import argparse

import config
from det_infer import DetInfer
import util 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TensorRT model")
    parser.add_argument("--dipoorlet", type=str, default=config.tensorrt_dipoorlet_model_path, help="Path to the TensorRT model")
    return parser.parse_args()

args = parse_args()

if args.dipoorlet:
    model_path = config.tensorrt_dipoorlet_model_path
else:
    model_path = config.tensorrt_model_path

# Load model, initialize inference engine
det_infer = DetInfer(model_path, "tensorrt")
det_infer.load_model(config.info)


with open(config.coco_anno_file, "r") as fp_gt:
    gt_data = json.load(fp_gt)

detection_out_dict = {
   "images": gt_data["images"],
   "annotations": [],
   "categories": gt_data["categories"]
}

# Load all COCO val images
coco = COCO(config.coco_anno_file)
image_ids = coco.getImgIds()
images = coco.loadImgs(image_ids)

t_start = time.time()
ann_idx = 0
for img_idx in range(len(images)):

    logger.info(img_idx)

    file_name = images[img_idx]["file_name"]
    img_path = os.path.join(config.coco_val_path, file_name)
    results, info = det_infer.inference(img_path, config.info)
    # logger.info(f"results : {results}")
    # logger.info(f"info : {info}")
    # det_infer.show_results_single_img(img_path, results, info, "/tmp/result.jpg")
    for result in results:
        detection_out_dict['annotations'].append(
            {
                "image_id": images[img_idx]["id"],
                "bbox": [
                    result[3],
                    result[4],
                    result[5],
                    result[6]
                ],
                "category_id": gt_data["categories"][result[0]]["id"],
                "id": ann_idx,
                "score": result[2],
                "area": result[5] * result[6]
            }
        )
        ann_idx += 1

t_end = time.time()
logger.info(f"tensorrt inference time: {t_end - t_start} for {len(images)} images.")

with open("./workdir/tensorrt_res.json", "w") as fp_out:
    json.dump(detection_out_dict, fp_out, ensure_ascii=False, indent=4)

util.print_coco_map(config.coco_anno_file, "./workdir/tensorrt_res.json")
