import os
import torch
import onnxruntime as ort 

# the path to save the onnx model
onnx_path = './weights/yolov8n.onnx'

torch_data = torch.randn([1, 3, 640, 640]).cuda()

print('{}: {:.1f} MiB'.format(onnx_path, os.path.getsize(onnx_path) / (2**20)))

from hidet.utils import benchmark_func

import numpy as np
import hidet

# load onnx model 'resnet50.onnx'
hidet_onnx_module = hidet.graph.frontend.from_onnx(onnx_path)

print('Input names:', hidet_onnx_module.input_names)
print('Output names: ', hidet_onnx_module.output_names)

# create a hidet tensor from pytorch tensor.
data: hidet.Tensor = hidet.from_torch(torch_data)

# apply the operators in onnx model to given 'data' input tensor
output: hidet.Tensor = hidet_onnx_module(data)

symbol_data = hidet.symbol_like(data)
symbol_output = hidet_onnx_module(symbol_data)
graph: hidet.FlowGraph = hidet.trace_from(symbol_output)
graph.save('./outs/graph_trace.hidet')

def bench_hidet_graph(graph: hidet.FlowGraph):
    cuda_graph = graph.cuda_graph()
    (output,) = cuda_graph.run([data])
    # np.testing.assert_allclose(
    #     actual=output.cpu().numpy(), desired=torch_output.cpu().numpy(), rtol=1e-2, atol=1e-2
    # )
    print(output.cpu().numpy().shape)
    print('  Hidet: {:.3f} ms'.format(benchmark_func(lambda: cuda_graph.run())))


bench_hidet_graph(graph)

# Set the search space level for kernel tuning. By default, the search space level is 0, which means no kernel tuning.
# There are three choices: 0, 1, and 2. The higher the level, the better performance but the longer compilation time.
hidet.option.search_space(0)

# optimize the flow graph, such as operator fusion
with hidet.graph.PassContext() as ctx:
    ctx.save_graph_instrument('./outs/graphs')
    graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)

bench_hidet_graph(graph_opt)
graph_opt.save('./outs/graph.hidet')


###
# graph_opt2 = hidet.graph.FlowGraph.load('./outs/graph.hidet')
# hidet.option.search_space(1)
# # optimize the flow graph, such as operator fusion
# with hidet.graph.PassContext() as ctx:
#     ctx.save_graph_instrument('./outs/graphs_opt2')
#     graph_opt2: hidet.FlowGraph = hidet.graph.optimize(graph_opt2)
# bench_hidet_graph(graph_opt2)
###