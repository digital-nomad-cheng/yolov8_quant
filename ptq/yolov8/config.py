
onnx_model_path = "/workspace/ptq/yolov8/weights/yolov8n.onnx"
tensorrt_model_path = "/workspace/ptq/yolov8/weights/yolov8n_int8.engine"
coco_val_path = "/workspace/ptq/datasets/coco2017/val2017/"
coco_anno_file = "/workspace/ptq/datasets/coco2017/annotations/instances_val2017.json"
dipoorlet_cali_dataset_path = "/workspace/ptq/yolov8/dipoorlet_work_dir/input.1/"

class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush')

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "output_shape": (1, 84, 8400),
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CUDAExecutionProvider"]
}
