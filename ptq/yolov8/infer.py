import os
from loguru import logger
from abc import abstractmethod, ABCMeta
import cv2

from backend import (
    load_onnx_model, run_onnx_model,
    load_tensorrt_model, run_tensorrt_model,
    load_tflite_model, run_tflite_model
)

class Infer(metaclass=ABCMeta):
    """
    Abstract Infer class for run combining different models and engines.
    """
    def __init__(self, model_path, backend):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            logger.info(f"model_path : {self.model_path}")
        else:
            raise Exception(f"{self.model_path} does not exist.")

        self.backend = backend
        if self.backend == "onnx":
            self.infer_model_func = run_onnx_model
            self.load_model_func = load_onnx_model
        elif self.backend == "tensorrt":
            self.infer_model_func = run_tensorrt_model
            self.load_model_func = load_tensorrt_model
        elif self.backend == "tflite":
            self.infer_model_func = run_tflite_model
            self.load_model_func = load_tflite_model
        else:
            raise Exception(f"Not supported {self.backend} backend.")

    def load_model(self, info):
        logger.info("Loading model...")     
        self.model, info = self.load_model_func(self.model_path, info)
        logger.info("Successfully loaded model.")
        return info

    def infer_model(self, inputs, info):
        return self.infer_model_func(inputs, self.model, info)

    @abstractmethod
    def preprocess(self, img_path, info):
        pass

    @abstractmethod
    def postprocess(self, outputs, info):
        pass

    def inference(self, img_path, info):
        inputs, info = self.preprocess(img_path, info)
        outputs, info = self.infer_model(inputs, info)
        results, info = self.postprocess(outputs, info)
        return results, info        
