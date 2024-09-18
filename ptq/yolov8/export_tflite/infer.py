import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import glob
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="saved_model/yolov8n_full_integer_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

# Function to preprocess the image
def preprocess_image(image_path, input_details):
    input_shape = input_details[0]['shape']
    image = Image.open(image_path)
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(image).astype(np.float32)
    
    image_array = (image_array / 255.0)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def quantize_input(image_array, input_details):
    scale, zero_point = input_details[0]['quantization']    
    image_array_int8 = (image_array / scale + zero_point).astype(np.int8)
    return image_array_int8

def inference(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Function to dequantize the output
def dequantize_output(output, output_details):
    scale, zero_point = output_details[0]['quantization']
    return (output.astype(np.float32) - zero_point) * scale

# Function to visualize the output (assuming object detection task)
def visualize_output(image_path, output):
    image = Image.open(image_path).resize((352, 352))
    draw = ImageDraw.Draw(image)
    
    draw = ImageDraw.Draw(image)
    
    # Apply NMS
    boxes = output[:, :4]
    scores = output[:, 4]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.4)
    
    for index in indices:
        i = index[0] if isinstance(index, np.ndarray) else index
        detection = output[i]
        if detection[4] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = detection[:4]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    return image

# Process all images in a directory
image_paths = glob.glob("/media/vincent/FAFC59F8FC59B01D/datasets/coco_minitrain_25k/images/train2017/*.jpg")[:100]
for image_path in image_paths:
    # Preprocess image
    image_array = preprocess_image(image_path, input_details)

    # Quantize input
    image_array_int8 = quantize_input(image_array, input_details)
    
    # Perform inference
    outputs = inference(image_array_int8)
    
    # Dequantize the output
    dequantized_outputs = dequantize_output(outputs, output_details)
    
    print(dequantized_outputs.shape)
    
    # Visualize output
    result_image = visualize_output(image_path, dequantized_outputs)
    
    # Save or display the result
    result_image.save(f"int8_output/output_{image_path.split('/')[-1]}_int8.jpg")
    # Alternatively, use result_image.show() to display the image
    
print("Inference and visualization complete.")