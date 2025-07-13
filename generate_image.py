from diffusers import StableDiffusionPipeline
import torch
import os

# Ensure this directory exists to store your model if you want it locally
model_id = "Linaqruf/anything-v4.0"

print("Loading model (this may take a few minutes on first run)...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    use_auth_token=True  # If you use a Hugging Face token
).to("cpu")

# Disable safety checker for anime models (optional)
pipe.safety_checker = None

# Example prompt (tweak to your liking)
prompt = "1girl, white dress, flower field, sky, flowing hair, anime style, detailed, soft light, masterpiece"
negative_prompt = "low quality, blurry, ugly, bad anatomy, bad hands"

# Generate image
print("Generating image...")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Save
output_path = "anime_image.png"
image.save(output_path)
print(f"Image saved to: {output_path}")
