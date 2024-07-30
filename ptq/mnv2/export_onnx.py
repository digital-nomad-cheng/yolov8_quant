import torch

def export_onnx(model_path="weights/mobilev2_model.pth"):
    sample_input = torch.randn(1, 3, 224, 224)
    model = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(model, torch.nn.DataParallel):
        print(model)
        model = model.module

    torch.onnx.export(model.cpu().eval(), sample_input, "weights/mnv2.onnx", export_params=True, do_constant_folding=True, opset_version=14)

if __name__ == "__main__":
    export_onnx()
