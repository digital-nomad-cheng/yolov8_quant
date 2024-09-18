# onnx2tf -i ../weights/yolov8n.onnx -oiqt -qt per-tensor
onnx2tf -i ../weights/yolov8n.onnx -oiqt -cind "images" "data/calibdata.npy" "[[[[0.0,0.0,0.0]]]]" "[[[[1.0,1.0,1.0]]]]" -qt per-tensor
