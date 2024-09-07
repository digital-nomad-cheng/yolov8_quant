import cv2
import numpy as np

from infer import Infer
import util

class DetInfer(Infer):
    def __init__(self, model_path, backend):
        super().__init__(model_path, backend)
        
    def preprocess(self, img_path, info):
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        info.update({"img_height": img_height, "img_width": img_width})
        img = util.letter_box(img, (info["input_width"], info["input_height"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return [img], info
    
    def postprocess(self, outputs, info):
        outputs = np.transpose(np.squeeze(outputs[0]))
        boxes = []
        scores = []
        class_ids = []
        for i in range(len(outputs)):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score > info["confidence_thres"]:
                x, y, w, h = outputs[i][:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                scores.append(max_score)
                class_id = np.argmax(classes_scores)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, info["confidence_thres"], info["iou_thres"])

        detections = []
        for i in indices:
            box = boxes[i]
            gain = min(info["input_width"] / info["img_width"], info["input_height"] / info["img_height"])
            pad = (
                round((info["input_width"] - info["img_width"] * gain) / 2 - 0.1),
                round((info["input_height"] - info["img_height"] * gain) / 2 - 0.1),
            )
            x1 = (box[0] - pad[0]) / gain
            y1 = (box[1] - pad[1]) / gain
            w = box[2] / gain
            h = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            detection = [class_id, info["class_names"][class_id], score.astype(np.float64), x1, y1, w, h]
            detections.append(detection)
        detections.sort(key=lambda x: x[2], reverse=True)
        if len(detections) > info["max_det"]:
            detections = detections[:info["max_det"]]
        return detections, info