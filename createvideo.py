import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageFilter
import numpy as np
import cv2
from pydub import AudioSegment
import time
import os
from tqdm import tqdm


class TextToImage:
    def __init__(self):
        self.default_negative_prompt = "lowres, bad anatomy, blurry, bad proportions, extra limbs, missing fingers, poorly drawn, signature, watermark, text, error, cropped, blurry face, extra limbs, fused hands, doll-like face, distorted anatomy, low resolution, low detail eyes, bad hands, distorted face"

        print("🔥 Initializing AI Model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "Meina/MeinaMix_V11",
            torch_dtype=torch.float32,
            safety_checker=None
        )
        self.pipe.enable_attention_slicing()
        self.pipe.to("cpu")
        # self.pipe.enable_sequential_cpu_offload()

    def create_image(self, prompt, width=768, height=512):
        styled_prompt = f"masterpiece, best quality, {prompt}, highly detailed, soft lighting, anime style, ultra sharp"

        image = self.pipe(
            styled_prompt,
            negative_prompt=self.default_negative_prompt,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # Post-processing
        # image = image.filter(ImageFilter.SHARPEN)
        return image

    def get_dimensions(self, aspect_ratio='16:9'):
        ratios = {
            '16:9': (768, 432),
            '9:16': (432, 768),
            '4:3': (640, 480),
            '1:1': (512, 512)
        }
        width, height = ratios.get(aspect_ratio, (512, 512))

        # Ensure divisible by 8 and ≥ 256
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)
        return width, height

def main():
    # ======================
    # CONFIGURATION
    # ======================
    AUDIO_FILE = "narration.mp3"  # Your pre-generated audio
    OUTPUT_FILE = "final_video.mp4"
    ASPECT_RATIO = "16:9"  # Options: "16:9", "9:16", "4:3", "1:1"

    # ======================
    # 1. INITIALIZE SYSTEMS
    # ======================
    tti = TextToImage()
    width, height = tti.get_dimensions(ASPECT_RATIO)

    # ======================
    # 3. SCENE DEFINITION
    # ======================
    scenes = [
        {
            "text": "An honest farmer had once an ass that had been a faithful servant to him a great many years, but was now growing old and unfit for work. The farmer, tired of keeping him, thought of putting an end to him. But the ass, sensing danger, quietly left and began a journey to the great city, hoping to become a musician.",
            "duration": 28.5,
            "start": 0.0,
            "end": 28.5,
            "prompt": "anime-style old donkey walking alone on a dirt path at dawn, autumn forest background, soft light, whimsical feel, emotional expression, fantasy setting"
        },
        {
            "text": "On the road, the ass met a tired dog lying by the roadside. The dog had run away because his master wanted to kill him for being too old to hunt. The ass invited him to join his journey to the city and try their luck as musicians.",
            "duration": 26.5,
            "start": 28.5,
            "end": 55.0,
            "prompt": "anime-style donkey talking to an old hound dog resting by a country road, warm atmosphere, overgrown grass, storytelling mood, expressive animal faces"
        },
        {
            "text": "A little farther on, they met a sad cat who had escaped drowning by her mistress for being too lazy to catch mice. The ass encouraged her to come with them to the city and become a night singer. The cat agreed and joined them.",
            "duration": 28.5,
            "start": 55.0,
            "end": 83.5,
            "prompt": "anime-style sad cat sitting in the road, speaking to donkey and dog, rustic countryside, fading daylight, hopeful emotion, detailed fur and eyes"
        },
        {
            "text": "Soon after, they heard a cock crowing loudly from a gate. He said he was to be killed for Sunday broth. The ass invited him to join them. The cock gladly agreed and the four set out joyfully toward the city.",
            "duration": 24.5,
            "start": 83.5,
            "end": 108.0,
            "prompt": "anime-style colorful rooster crowing on a wooden gate, surrounded by donkey, dog, and cat, early evening sky, joyful mood, bright tones, whimsical party of animals"
        },
        {
            "text": "As night fell, they took shelter in a forest. The donkey and dog slept under a tree, the cat climbed into the branches, and the cock perched at the very top. From there, he saw a distant light and alerted the group.",
            "duration": 28.5,
            "start": 108.0,
            "end": 136.5,
            "prompt": "anime-style nighttime forest, donkey and dog sleeping beneath a large tree, cat curled on branch, rooster high above spotting a distant light, moonlight glow, magical realism"
        },
        {
            "text": "They walked toward the light and discovered a house where a gang of robbers were feasting. The animals planned to scare them away and take over the house.",
            "duration": 22.5,
            "start": 136.5,
            "end": 159.0,
            "prompt": "anime-style group of animals peering through window into a robber hideout, candlelit feast, sneaky expressions, dark forest house, cozy interior lighting"
        },
        {
            "text": "The animals stacked on top of one another and began their loud music—braying, barking, meowing, and crowing—then crashed through the window, terrifying the robbers who fled into the night.",
            "duration": 26.5,
            "start": 159.0,
            "end": 185.5,
            "prompt": "anime-style dramatic moment: donkey, dog, cat, and rooster stacked and crashing through window, chaos inside, shocked robbers fleeing, magical spark effects"
        },
        {
            "text": "With the house empty, the animals enjoyed the feast left behind. Then they each found a place to sleep comfortably for the night.",
            "duration": 20.5,
            "start": 185.5,
            "end": 206.0,
            "prompt": "anime-style cozy cottage interior, animals eating at a wooden table, warm candlelight, rustic comfort, joyful expressions, soft glow"
        },
        {
            "text": "At midnight, one robber returned to scout the house. Mistaking the cat’s eyes for embers, he tried to light a match and was scratched. The dog bit him, the donkey kicked him, and the rooster crowed, sending him fleeing in terror.",
            "duration": 30.5,
            "start": 206.0,
            "end": 236.5,
            "prompt": "anime-style dramatic night scene: startled robber attacked by animals in a rustic kitchen, glowing cat eyes, chaotic action, dim lighting with sparks"
        },
        {
            "text": "The robber ran back, convinced the house was haunted by witches, monsters, and the devil. The robbers never returned. The animals, pleased with their new home, lived there happily ever after as musicians.",
            "duration": 130.5,
            "start": 236.5,
            "end": 367.0,
            "prompt": "anime-style peaceful countryside cottage at dawn, animals resting peacefully, rooster on rooftop, golden morning light, fairy tale ending, tranquil mood"
        }
    ]

    # ======================
    # 4. IMAGE GENERATION
    # ======================
    print("\n🖌️ Generating HD Frames...")
    for scene in tqdm(scenes):
        scene["image"] = tti.create_image(
            scene["prompt"],
            width=width,
            height=height
        )
        time.sleep(2)  # Cooling period

    # ======================
    # 5. VIDEO RENDERING
    # ======================
    def render_video():
        print("\n🎥 Frame-Perfect Video Assembly...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            "temp_video.mp4",
            fourcc,
            24,  # FPS
            (width, height)
        )

        prev_frame = None
        for scene in scenes:
            img = np.array(scene["image"])[:, :, ::-1].copy()
            frame_count = int((scene["end"] - scene["start"]) * 24)

            # Smooth transition
            if prev_frame is not None:
                for alpha in np.linspace(0, 1, 12):  # 0.5s fade
                    blended = cv2.addWeighted(prev_frame, 1 - alpha, img, alpha, 0)
                    video.write(blended)
                    frame_count -= 1

            # Main scene frames
            for _ in range(frame_count):
                video.write(img)
            prev_frame = img

        video.release()

    render_video()

    # ======================
    # 6. FINAL MERGE
    # ======================
    print("\n🔈 Synchronizing Audio...")
    os.system(f'ffmpeg -y -i temp_video.mp4 -i {AUDIO_FILE} '
              '-c:v libx264 -preset fast -crf 22 '
              '-c:a aac -b:a 192k -shortest '
              f'{OUTPUT_FILE}')
    os.remove("temp_video.mp4")

    print(f"\n✅ Perfectly Synced Video: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()