import torch
import ai_edge_torch



def export_tflite(model_path="weights/mobilev2_model.pth"):
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    model = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(model, torch.nn.DataParallel):
        print(model)
        model = model.module

    edge_model = ai_edge_torch.convert(model.cpu().eval(), sample_inputs)
    edge_model.export("weights/mnv2.tflite")


if __name__ == "__main__":
    export_tflite()
