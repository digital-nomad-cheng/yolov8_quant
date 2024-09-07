## MobileNetv2 on tiny-imagenet-200
1. Download tiny-imagenet dataset from:  http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Prepare the transformed dataset into binary format from numpy array, in `mnv2` directory, 
    run `prepare_cali_dataset.py`
3. Evaluate model performance under pytorch, run `eval_pytorch.py` 
4. Export onnx model, run `export_onnx.py`
5. With onnx model, export tensorrt model, run `export_tensorrt.py` with default quantization strategy
6. Evaluate tensorrt default quantization performance.
7. Use dirpoorlet to calibrate 
    ```
    python -m torch.distributed.launch --use_env -m dipoorlet -M MODEL_PATH -I INPUT_PATH -N PIC_NUM -A [mse, hist, minmax] -D [trt, snpe, rv, atlas, ti, stpu] [--bc] [--adaround] [--brecq] [--drop]
    ```
    + MSE Strategy
      ```
      python -m torch.distributed.launch --nproc_per_node 1 --use_env -m dipoorlet -I dipoorlet_work_dir/ -N 200 -D trt -M weights/mnv2.onnx -A mse -D trt -O mse_result
      ```
8. Write new trt engine file with generated parameters from dipoorlet
    ```
    python export_tensorrt_dipoorlet.py
    ```
9. Evaluate new trt engine performance
    | Strategy   | Acc   | Time for 10000 images |
    |------------|------------|------------------|
    | PyTorch | 67.5%|25.82s|
    | TensorRT KL int8| 64.88%|28.74s|
    | Dipoorlet MSE int8| 66.4%| |
    | Dipoorlet MSE+Brecq|
    
## YOLOv8n on coco-2017-val
1. Download COCO 2017 validation dataset:
   ```
   bash download_coco17_val.sh
   ```
2. Export yolov8n.onnx model file.
3. Evaluate onnx model performance on coco evaluation dataset.
   ```
   python eval_onnx.py
   ```
4. Build tensorrt engine file with default quantization strategy.
    ```
    python export_tensorrt.py
    ```
5. Evaluate tensorrt engine file performance.
    ```
    python eval_tensorrt.py
    ```
6. Table for performance comparison
    | Model | Backend | Quantization Method | MAP | Inference Time |
    |-------|---------|---------|----------------|-----------------|
    | YOLOv8n | ONNX | float32 | 0.35898 | 468.6s |
    | YOLOv8n | TensorRT | KL int8 | 0.31587 | 457.1s |
    | YOLOv8n | TensorRT | Brecq int8 | 0.35898 | 0.35898 |

