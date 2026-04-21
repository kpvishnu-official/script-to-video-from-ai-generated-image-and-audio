# Creating Sample image Generator class
 
from src.image.image_generator import ImageGenerator
import os

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)
# ---------------- PROMPT ---------------- #
prompt = (
    "1girl, brown hair, blue dress, sitting bored on grass, "
    "sunny meadow, soft sunlight, medium shot, centered composition"
)

# Small Model
gen = ImageGenerator(
    model_name="runwayml/stable-diffusion-v1-5",
    use_large_model=False
)
image_path = os.path.join(output_dir, "small_model.png")

result_path, final_prompt = gen.generate(
    prompt=prompt,
    output_path=image_path,
    seed=42  # optional (for consistency)
)
image_path = os.path.join(output_dir, "test_image.png")
# Medium Model
gen = ImageGenerator(
    model_name="Lykon/DreamShaper",
    use_large_model=True
)
image_path = os.path.join(output_dir, "medium_model.png")

result_path, final_prompt = gen.generate(
    prompt=prompt,
    output_path=image_path,
    seed=42  # optional (for consistency)
)

# Large Model
gen = ImageGenerator(
    model_name="Meina/MeinaMix_V11",
    use_large_model=True
)
image_path = os.path.join(output_dir, "large_model.png")

result_path, final_prompt = gen.generate(
    prompt=prompt,
    output_path=image_path,
    seed=42  # optional (for consistency)
)

