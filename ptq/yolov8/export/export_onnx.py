from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Export the model
model.export(format='onnx')

# Export tflite model
# This will also generate a onnx file but is different from above for tflite compatabilty
model.export(format='tflite', int8=True)
