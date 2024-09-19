import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="saved_model/yolov8n_float32.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# COCO class labels
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640))  # Resize to model input size
    image_array = np.array(image).astype(np.float32)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to perform inference
def inference(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def process_output(output, conf_threshold=0.4, iou_threshold=0.45):
    output = output[0].transpose((1, 0))  # Transpose to 8400x84
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
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold
    )
    
    return boxes[indices], class_ids[indices], confidences[indices]

# Function to visualize the output
def visualize_output(image_path, output):
    image = Image.open(image_path)
    original_size = image.size
    draw = ImageDraw.Draw(image)

    boxes, class_ids, confidences = process_output(output)
    for box, class_id, score in zip(boxes, class_ids, confidences):
        print(box)
        x1, y1, x2, y2 = box
        # Scale coordinates to original image size
        x1 = x1 * original_size[0] / 640
        y1 = y1 * original_size[1] / 640
        x2 = x2 * original_size[0] / 640
        y2 = y2 * original_size[1] / 640
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{COCO_CLASSES[int(class_id)]}: {score:.2f}", fill="red")
        
    return image

# Process all images in a directory
image_paths = glob.glob("/media/vincent/FAFC59F8FC59B01D/datasets/coco_minitrain_25k/images/train2017/*.jpg")[:100]
for image_path in image_paths:
    image_array = preprocess_image(image_path)
    outputs = inference(image_array)
    result_image = visualize_output(image_path, outputs)
    # result_image.show()  # Display the result
    result_image.save(f"float32_output/output_{image_path.split('/')[-1]}_float32.jpg")