import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb
from tflite import BuiltinOperator

import onnx
import tensorflow as tf


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

def get_tflite_quantizable_tensors(tflite_model_path):
    quantizable_tensors = []
    with open(tflite_model_path, 'rb') as f:
        model_buf = f.read()
        
    model_obj = schema_fb.Model.GetRootAsModel(model_buf, 0)
    subgraph = model_obj.Subgraphs(0)

    # Iterate over the operators (nodes in the model)
    for i in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(i)
        
        # Get operator code (i.e., the type of operation)
        opcode_index = operator.OpcodeIndex()
        operator_code = model_obj.OperatorCodes(opcode_index).BuiltinCode()
        layer_type = BuiltinCodeToName(operator_code)
        

        # Get input and output tensors for this operation
        inputs = [subgraph.Tensors(operator.Inputs(j)).Name().decode('utf-8') for j in range(operator.InputsLength())]
        outputs = [subgraph.Tensors(operator.Outputs(j)).Name().decode('utf-8') for j in range(operator.OutputsLength())]
        
        # Print the node (operation) details
        # print(f"Node {i}:")
        #print(f"Operation Code: {operator_code}")
        # print(f"Inputs: {inputs}")
        # print(f"Outputs: {outputs}")
        # print(f"Layer Type: {layer_type}")
        # print('-' * 40)
        if layer_type in ['CONV_2D', 'LOGISTIC', 'ADD', 'SUB', 'MUL', 'DIV', 'SOFTMAX', 'SIGMOID']:
            quantizable_tensors.append(i)
    return quantizable_tensors

def BuiltinCodeToName(code):
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None

        
def get_onnx_quantizable_tensors(onnx_model):
    all_node_types = set()
    for node in onnx_model.graph.node:
        # if node.op_type in ['Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'Sigmoid', 'Tanh']:
        all_node_types.add(node.op_type)
    print(all_node_types)
    
    quantizable_tensors = []
    quantizable_tensors.append(onnx_model.graph.input[0].name)
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type in ['Add', 'Sub', 'Mul', 'Div', 'Conv', 'Softmax', 'Sigmoid']:
            # 'Slice', 'Reshape', 'Concat', 'Transpose', 'MaxPool', 'Resize'
            for output in node.output:
                quantizable_tensors.append(output)
    quantizable_tensors.append(onnx_model.graph.output[0].name)
            # quantizable_nodes.append({"index": i, "node_name": node.name, "node_type": node.op_type})
    return quantizable_tensors
            
    
    
# Usage example
# tflite_model_path = "yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
# tensor_index = 0  # The index of the tensor you want to modify
# new_scale = 0.2
# new_zero_point = -2

# modify_tflite_quantization_params(tflite_model_path, tensor_index, new_scale, new_zero_point)

if __name__ == "__main__":
    onnx_model_path = "./weights/yolov8n.onnx"
    onnx_model = onnx.load(onnx_model_path)
    quantizable_tensors = get_onnx_quantizable_tensors(onnx_model)
    print(quantizable_tensors)
    print("Total onnx quantizable tensors: ", len(quantizable_tensors))
    
    tflite_model_path = "./export_tflite/yolov8n_saved_model/yolov8n_full_integer_quant.tflite"
    tflite_quantizable_tensors = get_tflite_quantizable_tensors(tflite_model_path)
    print(tflite_quantizable_tensors)
    print("Total tflite quantizable tensors: ", len(tflite_quantizable_tensors))    
    print(schema_fb.BuiltinOperator.__dict__.items())
    
    
    