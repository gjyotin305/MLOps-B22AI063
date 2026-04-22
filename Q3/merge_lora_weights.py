import torch
from diffusers import StableDiffusionPipeline

BASE_MODEL = "CompVis/stable-diffusion-v1-4"
LORA_PATH = "./sd-naruto-model-lora"
OUTPUT_DIR = "./sd-naruto-merged"
DEVICE = "cuda"

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(DEVICE)

# Load LoRA weights
pipe.load_lora_weights(LORA_PATH)

# Fuse LoRA into base model (IMPORTANT STEP)
pipe.fuse_lora()

# Optional: unload LoRA adapters after fusion
pipe.unload_lora_weights()

# Save merged model
pipe.save_pretrained(OUTPUT_DIR)

print(f"✅ Merged model saved at {OUTPUT_DIR}")