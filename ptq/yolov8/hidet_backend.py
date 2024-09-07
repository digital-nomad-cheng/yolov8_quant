import hidet
import torch

def load_hidet_model(model_path, info):
    model = hidet.graph.FlowGraph.load(model_path)
    return model, info

def run_hidet_model(inputs, model, info):
    cuda_graph = model.cuda_graph()
    hidet_inputs = [hidet.from_torch(torch.from_numpy(inp).cuda().contiguous()) for inp in inputs]
    (outputs,) = cuda_graph.run(hidet_inputs)
    return outputs.cpu().numpy(), info