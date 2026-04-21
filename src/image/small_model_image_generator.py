import gc
import logging
from typing import Tuple

import torch
from diffusers import StableDiffusionPipeline


def trim_prompt(prompt: str, max_words: int = 60) -> str:
    return " ".join(prompt.split()[:max_words])


class SmallModelImageGenerator:
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cpu",
        resolution: Tuple[int, int] = (512, 768)
    ) -> None:

        self.device: str = device
        self.resolution: Tuple[int, int] = resolution

        logging.info("🧠 Loading image generation model...")

        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        self.pipe.enable_attention_slicing()
        self.pipe.safety_checker = None

        # Prompt configs
        self.base_prompt: str = (
            "best quality, anime illustration, cinematic lighting, depth of field"
        )

        self.negative_prompt: str = (
            "blurry, low quality, bad anatomy, extra fingers, distorted, watermark, text"
        )

    def build_prompt(self, prompt: str) -> str:
        return f"{self.base_prompt}, {prompt}"

    def generate(
        self,
        prompt: str,
        output_path: str
    ) -> Tuple[str, str]:

        final_prompt: str = trim_prompt(self.build_prompt(prompt))

        logging.info("🖼️ Generating image...")

        with torch.no_grad():
            image = self.pipe(
                final_prompt,
                negative_prompt=self.negative_prompt,
                height=self.resolution[1],
                width=self.resolution[0],
                num_inference_steps=40,
                guidance_scale=8.0
            ).images[0]

        image.save(output_path)

        del image
        gc.collect()

        return output_path, final_prompt

    def cleanup(self) -> None:
        logging.info("🧹 Cleaning up model from memory...")
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
