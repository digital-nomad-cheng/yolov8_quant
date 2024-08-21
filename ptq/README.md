## MobileNetv2 on tiny-imagenet-200
1. Download tiny-imagenet dataset from:  http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Prepare the transformed dataset into binary format from numpy array, in `mnv2` directory, 
    run `prepare_cali_dataset.py`
3. Evaluate model performance under pytorch, run `eval_pytorch.py` 
4. Export onnx model, run `export_onnx.py`
5. With onnx model, export tensorrt model, run `export_tensorrt.py` with default quantization strategy
6. Evaluate tensorrt default quantization performance.



| Strategy   | Acc   | Time for 10000 images |
|------------|------------|------------------|
| PyTorch | 67.5%|25.82s|
| TensorRT| 64.88%|16.13s|
| Dipoorlet| | |

