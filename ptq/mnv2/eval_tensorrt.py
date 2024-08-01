import time

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch

from dataset import get_dataset

logger = trt.Logger(trt.Logger.VERBOSE)

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __reprt__(self):
        return self.__str__()

def load_engine(engile_file):
    runtime = trt.Runtime(logger)
    with open(engile_file, "rb") as f:
        engine = f.read()
    return runtime.deserialize_cuda_engine(engine)

def allocate_buffers(engine):
    """
    Allocate paged host memory and cuda memory
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def infer(
    context, 
    bindings,
    inputs, 
    outputs, 
    stream,
    batch_size=1
):
    """
    Perform inference and transfer data between host and device.
    """
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return outputs

def eval_tensorrt_int8(
    engine_file="weights/mnv2_int8.engine"
):
    _, val_dataset, _ = get_dataset()
    data_loaders = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=8,
    )
    engine = load_engine(engine_file)
    context = engine.create_execution_context()
    wrapper_inputs, wrapper_outputs, bindings, stream =  allocate_buffers(engine)
    
    output_shape = (1, 200)
    running_corrects = 0.0
    t_start = time.time()
    for i, (inputs, labels) in enumerate(data_loaders):
        inputs = inputs.numpy()
        wrapper_inputs[0].host = inputs.reshape(-1)
        wrapper_outputs = infer(context, bindings=bindings, inputs=wrapper_inputs, outputs=wrapper_outputs, stream=stream)
        output = wrapper_outputs[0].host.reshape(*output_shape)
        output = torch.tensor(output)
        _, preds = torch.max(output, 1)
        running_corrects += torch.sum(preds == labels.data)
    t_end = time.time()
    print(f"Total inference time for {len(val_dataset)} images is {t_end - t_start} seconds")
    print(f"Acc with TRT int8 infer: {running_corrects / len(val_dataset) * 100}%")

if __name__ == "__main__":
    eval_tensorrt_int8()
