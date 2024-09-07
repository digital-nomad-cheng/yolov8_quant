import numpy as np
import hidet
import torch

from hidet.utils import benchmark_func

onnx_path = './weights/yolov8n.onnx'
torch_data = torch.randn([1, 3, 640, 640]).cuda()

hidet_onnx_module = hidet.graph.frontend.from_onnx(onnx_path)
print('Input names:', hidet_onnx_module.input_names)
print('Output names: ', hidet_onnx_module.output_names)

data: hidet.Tensor = hidet.from_torch(torch_data)

output: hidet.Tensor = hidet_onnx_module(data)

symbol_data = hidet.symbol_like(data)
symbol_output = hidet_onnx_module(symbol_data)
graph: hidet.FlowGraph = hidet.trace_from(symbol_output)


hidet.option.search_space(0)
with hidet.graph.PassContext() as ctx:
    ctx.save_graph_instrument('./outs/graphs')
    graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)

graph_opt.save('./weights/yolov8n.hidet')
