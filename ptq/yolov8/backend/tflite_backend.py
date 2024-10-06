import tensorflow as tf
import numpy as np

# Function to quantize the input
def quantize_input(image_array, input_details, dtype=np.int8):
    scale, zero_point = input_details[0]['quantization']    
    image_array_int = (image_array / scale + zero_point).astype(dtype)
    return image_array_int

# Function to dequantize the output
def dequantize_output(output, output_details):
    scale, zero_point = output_details[0]['quantization']
    return (output.astype(np.float32) - zero_point) * scale

def load_tflite_model(model_path, info):
    interpreter = tf.lite.Interpreter(model_path=model_path) 
    interpreter.allocate_tensors()
    info["input_details"] = interpreter.get_input_details()
    info["output_details"] = interpreter.get_output_details()
    return interpreter, info

def run_tflite_model(inputs, interpreter, info):
    inputs = quantize_input(inputs[0], info["input_details"])
    interpreter.set_tensor(info["input_details"][0]["index"], inputs)
    interpreter.invoke()
    outputs = interpreter.get_tensor(info["output_details"][0]["index"])
    outputs = dequantize_output(outputs, info["output_details"])
    return outputs, info
