import json
import numpy as np

def calculate_scale_zero_point(min_val, max_val, num_bits=8):
    # qmin = 0.
    # qmax = 2.**num_bits - 1.
    qmin = float(-2**(num_bits - 1))
    qmax = float(2**(num_bits - 1) - 1)
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = np.clip(initial_zero_point, qmin, qmax)
    zero_point = int(round(zero_point))
    return scale, zero_point

# Load the activation clip values
with open("dipoorlet_work_dir/yolov8n_mse/act_clip_val.json", "r") as f:
    act_clip_val = json.load(f)

quant_params = {}

for layer_name, clip_values in act_clip_val.items():
    min_val, max_val = clip_values
    scale, zero_point = calculate_scale_zero_point(min_val, max_val)
    
    quant_params[layer_name] = {
        "scale": float(scale),
        "zero_point": zero_point,
        "min": min_val,
        "max": max_val
    }

# Write quantization parameters to a new JSON file
with open("scale_and_zeropoint_params.json", "w") as f:
    json.dump(quant_params, f, indent=2)

print("Activation quantization parameters have been written to scale_and_zeropoint_params.json")
