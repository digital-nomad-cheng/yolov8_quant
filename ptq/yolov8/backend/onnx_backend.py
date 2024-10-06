import onnxruntime as ort


def load_onnx_model(model_path, info):
    model = ort.InferenceSession(model_path, providers=info["providers"])
    return model, info

def run_onnx_model(inputs, model, info):
    outputs_name = info["outputs_name"]
    inputs_name = info["inputs_name"]
    inputs_dict = dict(zip(inputs_name, inputs))
    outputs = model.run(outputs_name, inputs_dict)
    return outputs, info
