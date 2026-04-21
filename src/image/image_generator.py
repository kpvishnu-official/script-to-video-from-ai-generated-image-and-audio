import gc
import logging
from typing import Tuple, Optional

import torch
from diffusers import StableDiffusionPipeline


def trim_prompt(prompt: str, max_words: int = 60) -> str:
    return " ".join(prompt.split()[:max_words])


class ImageGenerator:
    def __init__(
        self,
        model_name: str = "Lykon/DreamShaper",
        device: str = "cpu",
        resolution: Tuple[int, int] = (512, 768),
        use_large_model: bool = False,
        # --- Consistency options ---
        character_description: Optional[str] = None,
        scene_description: Optional[str] = None,
        fixed_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            character_description:
                A detailed, stable description of your main character that will
                be prepended to EVERY prompt. This is the single most effective
                way to keep the same face/look across scenes without training.
                Example:
                    "young woman, long auburn hair, green eyes, pale skin,
                     wearing a dark blue cloak"

            scene_description:
                Persistent environment/mood details prepended to every prompt.
                Example:
                    "medieval fantasy village, warm candlelight, foggy cobblestone streets"

            fixed_seed:
                When set, all images use the same RNG seed. Combined with a
                strong character_description this greatly improves consistency.
                Set to None to use a random seed per image.
        """
        self.device: str = device
        self.resolution: Tuple[int, int] = resolution
        self.use_large_model: bool = use_large_model
        self.character_description: Optional[str] = character_description
        self.scene_description: Optional[str] = scene_description
        self.fixed_seed: Optional[int] = fixed_seed

        logging.info(f"🧠 Loading model: {model_name}")
        dtype = torch.float32  # CPU safe

        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)

        self.pipe.enable_attention_slicing()
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass
        self.pipe.safety_checker = None

        # Style anchor — applied to every image
        self.style_prompt: str = (
            "best quality, anime illustration, cinematic lighting, "
            "depth of field, sharp eyes, consistent art style"
        )

        # Negative prompt — prevents drift in quality and anatomy
        self.negative_prompt: str = (
            "blurry, low quality, bad anatomy, extra fingers, distorted, "
            "watermark, text, different art style, inconsistent style, "
            "multiple people, wrong face, face change"
        )

        if self.use_large_model:
            logging.info("⚠️ Large model mode enabled (slower but better quality)")
            self.steps = 50
            self.guidance = 8.5
        else:
            self.steps = 35
            self.guidance = 8.0

        if self.fixed_seed is not None:
            logging.info(f"🔒 Fixed seed: {self.fixed_seed} — character consistency enabled")
        if self.character_description:
            logging.info(f"🧍 Character anchor: '{self.character_description[:60]}...'")
        if self.scene_description:
            logging.info(f"🌄 Scene anchor: '{self.scene_description[:60]}...'")

    # ---------------- PROMPT BUILDING ---------------- #
    def build_prompt(self, scene_prompt: str) -> str:
        """
        Assembles the final prompt in order of importance:
          1. Character description  (most important for face/look consistency)
          2. Scene/environment      (keeps world consistent)
          3. Scene-specific action  (what's happening this moment)
          4. Style anchor           (quality + art style lock)

        Stable Diffusion weighs earlier tokens more heavily, so
        character details must come first.
        """
        parts = []

        if self.character_description:
            parts.append(self.character_description)

        if self.scene_description:
            parts.append(self.scene_description)

        parts.append(scene_prompt)
        parts.append(self.style_prompt)

        return trim_prompt(", ".join(parts), max_words=70)

    # ---------------- GENERATE ---------------- #
    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Args:
            prompt:      The scene-specific prompt for this image.
            output_path: Where to save the PNG.
            seed:        Per-call seed override. Falls back to fixed_seed,
                         then to random if both are None.
        """
        final_prompt: str = self.build_prompt(prompt)
        logging.info(f"🖼️  Generating image — prompt: '{final_prompt[:80]}...'")

        # Seed priority: explicit per-call → fixed_seed → random
        active_seed = seed if seed is not None else self.fixed_seed
        generator = None
        if active_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(active_seed)

        with torch.no_grad():
            image = self.pipe(
                final_prompt,
                negative_prompt=self.negative_prompt,
                height=self.resolution[1],
                width=self.resolution[0],
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=generator,
            ).images[0]

        image.save(output_path)
        logging.info(f"💾 Saved: {output_path}")

        del image
        gc.collect()

        return output_path, final_prompt

    # ---------------- CLEANUP ---------------- #
    def cleanup(self) -> None:
        logging.info("🧹 Cleaning up model from memory...")
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
