import time
import os
from diffusers import OnnxStableDiffusionPipeline

# MODEL_ID = "onnx-community/stable-diffusion-v1-5-ONNX"
# ONNX_DIR = "./onnx_model"  # where model gets cached

# from diffusers import OnnxStableDiffusionPipeline
# height=512
# width=512
# num_inference_steps=50
# guidance_scale=7.5
# prompt = "a photo of an astronaut riding a horse on mars"
# negative_prompt="bad hands, blurry"
pipe = OnnxStableDiffusionPipeline.from_pretrained("./sd-naruto-onnx", provider="CUDAExecutionProvider")

# image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 
# image.save("astronaut_rides_horse.png")

pipe.set_progress_bar_config(disable=True)

prompt = "a photo of an astronaut riding a horse on mars"

# -------------------------
# LATENCY BENCHMARK
# -------------------------
times = []

for i in range(5):
    start = time.time()

    image = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=10,
        num_images_per_prompt=1
    ).images[0]

    end = time.time()
    latency = end - start
    times.append(latency)

    print(f"Run {i+1}: {latency:.2f} sec")

    if i == 0:
        image.save("output.png")  # save once

avg_time = sum(times) / len(times)

print("\n===== RESULT =====")
print(f"Average inference time (Python ONNX): {avg_time:.2f} sec")


def get_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024**3)


# HuggingFace cache path
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

onnx_size = get_size_gb(cache_dir)

print(f"ONNX model size (approx): {onnx_size:.2f} GB")