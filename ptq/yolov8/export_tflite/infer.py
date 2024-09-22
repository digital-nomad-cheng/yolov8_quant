import tensorflow as tf
from config import COCO_CLASSES
from PIL import Image
import numpy as np
import cv2
from PIL import ImageDraw
import glob
import os
import utils
from tqdm import tqdm


class Infer:
    def __init__(self, saved_model_path, mode="int8",input_sizes=(640, 640)):
        if mode == "float32":
            model_path = os.path.join(saved_model_path, "yolov8n_float32.tflite")
        elif mode == "int8":
            model_path = os.path.join(saved_model_path, "yolov8n_full_integer_quant.tflite")
        else:
            model_path = os.path.join(saved_model_path, "yolov8n_full_integer_quant_with_int16_act.tflite")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_sizes = input_sizes
        self.mode = mode

    def infer(self, image_path, visualize=False):
        image_array = self.preprocess_image(image_path)
        
        # quantize input if it's int8 or int16
        if self.mode == "int8":
            image_array = self.quantize_input(image_array, self.input_details)
        elif self.mode == "int16":
            image_array = self.quantize_input(image_array, self.input_details, dtype=np.int16)
            
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # dequantize output if it's int8 or int16
        if self.mode == "int8":
            output = self.dequantize_output(output, self.output_details)
        elif self.mode == "int16":
            output = self.dequantize_output(output, self.output_details)
            
        boxes, class_ids, confidences = self.process_output(output)

        return boxes, class_ids, confidences
    
    # Function to preprocess the image
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        self.original_size = image.size
        image = image.resize(self.input_sizes)  # Resize to model input size
        image_array = np.array(image).astype(np.float32)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    def process_output(self, output, conf_threshold=0.2, iou_threshold=0.4):
        output = output[0].transpose((1, 0))  # Transpose to 8004x84
        boxes = output[:, :4]  # x, y, w, h
        scores = output[:, 4:]  # class scores
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Convert to corner coordinates
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0] = boxes[:, 0] * self.original_size[0]
        boxes[:, 1] = boxes[:, 1] * self.original_size[1]
        boxes[:, 2] = boxes[:, 2] * self.original_size[0]
        boxes[:, 3] = boxes[:, 3] * self.original_size[1]
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold
        )
        
        return boxes[indices], class_ids[indices], confidences[indices]
    
    def quantize_input(self, image_array, input_details, dtype=np.int8):
        scale, zero_point = input_details[0]['quantization']    
        image_array_int = (image_array / scale + zero_point).astype(dtype)
        return image_array_int
    
    # Function to dequantize the output
    def dequantize_output(self, output, output_details):
        scale, zero_point = output_details[0]['quantization']
        return (output.astype(np.float32) - zero_point) * scale
    
    # Function to visualize the output
    def visualize_output(self, image_path, boxes, class_ids, confidences, save_img=False):
        image = Image.open(image_path)
        original_size = image.size
        draw = ImageDraw.Draw(image)

        for box, class_id, score in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{COCO_CLASSES[int(class_id)]}: {score:.2f}", fill="red")
        
        if save_img:
            image.save(f"{self.mode}_output/output_{image_path.split('/')[-1]}")
        else:
            image.show()
    
if __name__ == "__main__":
    infer = Infer("onnx2tf_yolov8n_saved_model", mode="int8")
    # Process all images in a directory
    image_paths = glob.glob("/media/vincent/FAFC59F8FC59B01D/datasets/coco16/images/train2017/*.jpg")
    annotations_file = "/media/vincent/FAFC59F8FC59B01D/datasets/coco16/instances_train2017.json"
    
    coco = utils.load_coco_annotations(annotations_file)
    all_detections = []
    for image_path in tqdm(image_paths):
        image_id = int(os.path.basename(image_path).split('.')[0])
        boxes, class_ids, confidences = infer.infer(image_path, visualize=True)
        infer.visualize_output(image_path, boxes, class_ids, confidences, save_img=True)
        coco_detections = utils.convert_to_coco_format(image_id, boxes, class_ids, confidences)
        all_detections.extend(coco_detections)
    
    metrics = utils.calculate_coco_metrics(coco, all_detections)
    print(metrics)