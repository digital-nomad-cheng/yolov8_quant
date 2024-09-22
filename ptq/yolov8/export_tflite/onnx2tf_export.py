import onnx2tf

f_onnx = "./yolov8n.onnx"
np_data = [["images", "data/calibdata.npy", [[[[0,0,0]]]],[[[[1,1,1]]]]]]

onnx2tf.convert(
    input_onnx_file_path=f_onnx,
    output_folder_path="onnx2tf_yolov8n_saved_model",
    not_use_onnxsim=True,
    verbosity="info",
    output_integer_quantized_tflite=True,
    quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
    custom_input_op_name_np_data_path=np_data,
    disable_group_convolution=True,  # for end-to-end model compatibility
    enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    batch_size=1,
)
