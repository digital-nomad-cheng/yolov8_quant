from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_to_coco_format(image_id, boxes, class_ids, confidences, categories):
    coco_detections = []
    for box, class_id, score in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        coco_detections.append({
            "image_id": image_id,
            "category_id": categories[int(class_id)]["id"],
            "bbox": [x1, y1, width, height],
            "score": float(score)
        })
    return coco_detections

def load_coco_annotations(annotation_file):
    coco = COCO(annotation_file)
    return coco

def calculate_coco_metrics(coco, detections):
    coco_dt = coco.loadRes(detections)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats