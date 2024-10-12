## MobileNetv2 on tiny-imagenet-200 with TensorRT
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
    
## YOLOv8n on coco-2017-val with TensorRT
1. Download COCO 2017 validation dataset:
   ```
   bash download_coco17_val.sh
   ```
2. Export yolov8n.onnx model file.
    ```
    # Note this script will use the intermediate onnx file generated when exporting tflite file
    python export_onnx.py 
    ```
3. Evaluate onnx model performance on coco evaluation dataset.
   ```
   python eval_onnx.py
   # Results
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.291
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.580
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
   ```
4. Build tensorrt engine file with default quantization strategy.
    ```
    python export/export_tensorrt.py
    ```
5. Evaluate tensorrt engine file performance.
    ```
    python eval_tensorrt.py
    # Results
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.314
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.455
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.343
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.346
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.265
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.483
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.257
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
    ```
6. Use dirpoorlet to calibrate 
    ```
    # use mse as calibration strategy
    python -m torch.distributed.launch --nproc_per_node 1 --use_env -m dipoorlet -I dipoorlet_work_dir/ -N 1024 -D trt -M weights/yolov8n.onnx -A mse -D trt -O yolov8n_mse
    # use brecq 
    python -m torch.distributed.launch --nproc_per_node 1 --use_env -m dipoorlet -I dipoorlet_work_dir/input.1/ -N 100 -D trt -M weights/yolov8n.onnx -A mse --brecq -D trt -O yolov8n_mse_brecq_n100
    ```
    We can see from the generated file in work_dir, there are two files that will be useful for us. The first one is `act_clip_val.json` which is the optimal activation clipping range calcuated with the calibration strategy.
    The other one is `trt_clip_val.json`, since tensorrt is using symmetric quantization strategy, it's using the `max(abs(min_act_val), abs(max_act_val))` in `act_clip_val.json` for each layer.
    We will use this file in next step to change the quantization parameters of tensorrt.

7. Write new trt engine file with generated parameters from dipoorlet. Since tensorrt uses symmetric quantization strategy for activation, we can see that the absolute values of dynamic range are the same for each layer.
   After we set the dynamic range of tensorrt, tensorrt will use these values to calcuate new scales which will lead to better performance.
    ```
    python export/export_tensorrt_dipoorlet.py
    ```
8. Evaluate new trt engine performance
    ```
    # change the tensorrt engine file to the new generated with dipoorlet dynamic range
    python eval_tensorrt.py
    # results
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.500
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.392
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.503
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.473
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.519
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
    ```
6. Table for performance comparison, due to the long time of calibration, I only use 100 images for calibration to produce the table below. 
    Time is measured on 5000 evaluation images.
    | Model | Backend | Quantization Method | MAP | Inference Time |
    |-------|---------|---------|----------------|-----------------|
    | YOLOv8n | ONNX | float32 | 0.35898 | 356.8s |
    | YOLOv8n | TensorRT | KL int8 | 0.31587 | 329.0s |
    | YOLOv8n | TensorRT | MSE int8 | 0.35286 | 330.2s |

    We can see from the table that even with 100 images, the MSE calibration strategy can already improve the performance of the model that than the default KL strategy.
    With more calibration images, the performance of MSE strategy will be even better.

## Reference
1. onnxruntime-gpu cannot find CUDAProvider: install onnxruntime-gpu from:
    ```
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    ```
