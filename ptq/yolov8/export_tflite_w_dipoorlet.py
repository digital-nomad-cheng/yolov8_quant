import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

def modify_tflite_quantization_params(model_path, tensor_index, new_scale, new_zero_point):
    # Read the model file
    with open(model_path, 'rb') as f:
        model_buf = f.read()

    # Get model object
    model = schema_fb.Model.GetRootAsModel(model_buf, 0)

    # Create a mutable copy of the model
    model_obj = schema_fb.ModelT.InitFromObj(model)

    # Find the tensor we want to modify
    tensor = model_obj.subgraphs[0].tensors[tensor_index]
    print(tensor.name)
    # Modify quantization parameters
    if tensor.quantization is None:
        tensor.quantization = schema_fb.QuantizationParametersT()
    
    tensor.quantization.scale = [new_scale]
    tensor.quantization.zeroPoint = [new_zero_point]

    # Create a new FlatBuffer builder
    builder = flatbuffers.Builder(0)

    # Serialize the modified model
    model_buf = model_obj.Pack(builder)
    builder.Finish(model_buf)

    # Get the modified model as bytes
    modified_model_buf = builder.Output()

    # Write the modified model to a file
    output_path = model_path.replace('.tflite', '_modified.tflite')
    with open(output_path, 'wb') as f:
        f.write(modified_model_buf)

    print(f"Modified model saved to {output_path}")

# Usage example
model_path = "yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
tensor_index = 0  # The index of the tensor you want to modify
new_scale = 0.2
new_zero_point = -2

modify_tflite_quantization_params(model_path, tensor_index, new_scale, new_zero_point)
