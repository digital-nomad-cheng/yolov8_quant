import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
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

def load_tensorrt_model(model_path, info):
    runtime = trt.Runtime(LOGGER)
    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine, info

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def run_tensorrt_model(inputs, model, info):
    shape_of_output = info["output_shape"]
    context = model.create_execution_context()
    trt_inputs, trt_outputs, bindings, stream = allocate_buffers(model)
    trt_inputs[0].host = inputs[0].reshape(-1)
    outputs = do_inference(context, bindings=bindings, inputs=trt_inputs, outputs=trt_outputs, stream=stream)
    feat = postprocess_the_outputs(outputs[0], shape_of_output)
    return feat, info
