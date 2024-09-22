from ultralytics import YOLO

model = YOLO("yolov8n.pt")
path = model.export(format="tflite", int8=True)
# the onnx model get above is different from the code below
# path = model.export(format="onnx")

print(path)