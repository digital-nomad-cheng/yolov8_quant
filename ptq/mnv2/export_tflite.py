import glob

from PIL import Image
import numpy as np
import torch
import ai_edge_torch
import tensorflow as tf

def cali_dataset_gen():
    print("total calibration imgs:", len(glob.glob("../tiny-imagenet-200/test/images/test_9*.JPEG")))
    for img_file in glob.glob("../tiny-imagenet-200/test/images/9*.JPEG"):
        img = Image.open(img_file).convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
        yield [img]

def get_sample_dataset():
    print("total calibration imgs:", len(glob.glob("../tiny-imagenet-200/test/images/test_9*.JPEG")))
    imgs = []
    for img_file in glob.glob("../tiny-imagenet-200/test/images/9*.JPEG"):
        img = Image.open(img_file).convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
        imgs.append(img)

def export_tflite(model_path="weights/mobilev2_model.pth"):
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    model = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(model, torch.nn.DataParallel):
        print(model)
        model = model.module

    edge_model = ai_edge_torch.convert(model.cpu().eval(), sample_inputs)
    edge_model.export("weights/mnv2.tflite")

def export_quantized_tflite(model_path="weights/mobilev2_model.pth"):
    supported_types = {"supported_types": [tf.int8]}
    tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT], 
                           'target_spec': supported_types,
                           "representative_dataset": cali_dataset_gen,
                           "inference_input_type": tf.int8,
                           "inference_output_type": tf.int8}

    sample_inputs = (torch.randn(1, 3, 224, 224),)
    model = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(model, torch.nn.DataParallel):
        print(model)
        model = model.module

    edge_model = ai_edge_torch.convert(model.cpu().eval(), sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
    edge_model.export("weights/mnv2_quantized.tflite")

if __name__ == "__main__":
    # export_tflite()
    export_quantized_tflite()
