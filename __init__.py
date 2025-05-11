import torch
import math
import logging
import numpy
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

def sigmoid_offset_scheduler(model_sampling, steps: int, square_k: float = 1.0, base_c: float = 0.5) -> torch.Tensor:

    if not hasattr(model_sampling, 'sigmas') or len(model_sampling.sigmas) < 2:
        logging.error("model_sampling object must have a 'sigmas' attribute with at least two values.")
        return torch.FloatTensor([1.0, 0.0])

    if steps <= 0:
        logging.warning(f"Sigmoid scheduler called with steps={steps}. Returning minimal schedule.")
        sigma_max_val = float(model_sampling.sigmas[-1])
        return torch.FloatTensor([sigma_max_val, 0.0])

    total_timesteps = (len(model_sampling.sigmas) -1)
    
    # Generate normalized sequence from 0 to 1 (exclusive)
    ts = numpy.linspace(0, 1, steps, endpoint=False)
    
    # Apply sigmoid transformation
    # Normalize base_c to center around 0 for shifting
    shift = 2.0 * (base_c - 0.5)  # Maps 0-1 to -1 to 1
    
    def sigmoid(x, k=square_k, shift=shift):
        # Convert from [0,1] to [-4,4] range for good sigmoid coverage
        x = 8.0 * x - 4.0
        # Apply shift from base_c
        x = x + (shift * 4.0)  # Scale shift by width of range
        
        # Apply sigmoid with steepness k
        if k * x > 700:
            return 1.0
        elif k * x < -700:
            return 0.0
        else:
            return 1.0 / (1.0 + numpy.exp(-k * x))
    
    # Transform using sigmoid and flip (1-x) for correct mapping
    transformed_ts = 1.0 - numpy.array([sigmoid(t) for t in ts])
    
    # Map to timesteps in the model_sampling.sigmas range
    mapped_ts = numpy.rint(transformed_ts * total_timesteps).astype(int)
    
    # Collect sigma values
    sigs = []
    last_t = -1
    for t in mapped_ts:
        if t != last_t:
            if isinstance(model_sampling.sigmas, torch.Tensor):
                sigs.append(float(model_sampling.sigmas[t].item()))
            else:  # Assume list or numpy array
                sigs.append(float(model_sampling.sigmas[t]))
        last_t = t
    
    # Add final 0.0
    sigs.append(0.0)
    
    return torch.FloatTensor(sigs)

new_scheduler_name = "sigmoid_offset"
if new_scheduler_name not in SCHEDULER_HANDLERS:
    sigmoid_offset_handler = SchedulerHandler(handler=sigmoid_offset_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[new_scheduler_name] = sigmoid_offset_handler
    SCHEDULER_NAMES.append(new_scheduler_name)


class SigmoidOffsetScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                     "square_k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step":0.01, "round": False, "tooltip": "Sigmoid steepness factor. Higher values = steeper transition."}),
                     "base_c": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step":0.01, "round": False, "tooltip": "Control parameter (0.0-1.0) that shifts sigmoid curve. '< 0.5: More time at low sigmas (early denoising)' and '> 0.5: More time at high sigmas (late denoising)'."}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, square_k, base_c):
        sigmas = sigmoid_offset_scheduler(model.get_model_object("model_sampling"), steps, square_k=square_k, base_c=base_c)
        return (sigmas, )



NODE_CLASS_MAPPINGS = {
    "SigmoidOffsetScheduler": SigmoidOffsetScheduler,
}
