import torch
import os
from diffusers import StableDiffusionPipeline

BASE_MODEL = "CompVis/stable-diffusion-v1-4"
LORA_PATH = "./sd-naruto-model-lora"  # folder or .bin/.safetensors
DEVICE = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


unet_params = count_params(pipe.unet)
vae_params = count_params(pipe.vae)
text_encoder_params = count_params(pipe.text_encoder)

base_total = unet_params + vae_params + text_encoder_params

print("\n=== BASE MODEL PARAMS ===")
print(f"UNet: {unet_params:,}")
print(f"VAE: {vae_params:,}")
print(f"Text Encoder: {text_encoder_params:,}")
print(f"TOTAL: {base_total:,}")


pipe.unet.load_attn_procs(LORA_PATH)

lora_params = 0

for name, module in pipe.unet.named_modules():
    if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
        for param in module.parameters():
            if param.requires_grad:
                lora_params += param.numel()

print("\n=== LORA PARAMS ===")
print(f"Trainable LoRA params: {lora_params:,}")

combined_total = base_total + lora_params
print(f"Combined params: {combined_total:,}")


def get_folder_size_mb(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

lora_size = get_folder_size_mb(LORA_PATH)

print("\n=== LORA FILE SIZE ===")
print(f"LoRA size: {lora_size:.2f} MB")

# =========================
# INFERENCE
# =========================
prompt = "Naruto Uzumaki, anime style, orange outfit, glowing blue chakra, high detail"

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("output.png")

print("\n✅ Image saved as output.png")

import time

# =========================
# BENCHMARK SETTINGS
# =========================
num_runs = 5
steps = 20

# Warmup (very important for GPU timing)
_ = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
torch.cuda.synchronize()

print("\n=== BENCHMARKING ===")

times = []

for i in range(num_runs):
    start = time.time()

    _ = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=7.5
    ).images[0]

    torch.cuda.synchronize()  # ensure accurate GPU timing
    end = time.time()

    run_time = end - start
    times.append(run_time)
    print(f"Run {i+1}: {run_time:.4f} sec")

avg_time = sum(times) / len(times)

print("\n=== RESULTS ===")
print(f"Average inference time ({num_runs} runs, {steps} steps): {avg_time:.4f} sec")
print(f"Images/sec: {1/avg_time:.4f}")