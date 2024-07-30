import tensorrt as trt
from prepare_cali_dataset import TRTCalibrator

logger = trt.Logger(trt.Logger.VERBOSE)

def export_trt_engine(
    onnx_file="weights/mnv2.onnx",
    engine_file="weights/mnv2_int8.engine",
    
):
    trt_builder = trt.Builder(logger)
    trt_net = trt_builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    trt_parser = trt.OnnxParser(trt_net, logger)
    trt_parser.parse_from_file(onnx_file)
    
    trt_config = trt_builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))
    trt_config.set_flag(trt.BuilderFlag.INT8)
    trt_config.int8_calibrator = TRTCalibrator() 

    trt_engine = trt_builder.build_serialized_network(trt_net, trt_config)
    
    if trt_engine is None:
        print("Failed to export tensorrt engine.")
        exit(-1)

    with open(engine_file, "wb") as f:
        f.write(trt_engine)


if __name__ == "__main__":
    export_trt_engine()
