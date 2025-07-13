import json
import subprocess
from diffusers import StableDiffusionPipeline
import torch

import os
import asyncio
import edge_tts

class TextToVideoCreator:
    def __init__(self, json_path, output_dir="media", resolution=(720, 1280)):
        self.json_path = json_path
        self.output_dir = output_dir
        self.resolution = resolution
        self.prompts = self.load_json()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/video", exist_ok=True)

        self.device = "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "Meina/MeinaMix_V11", torch_dtype=torch.float32
        ).to(self.device)

    def load_json(self):
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def generate_image(self, prompt, index):
        image = self.pipe(prompt, height=self.resolution[1], width=self.resolution[0]).images[0]
        img_path = f"{self.output_dir}/images/frame_{index}.png"
        image.save(img_path)
        return img_path

    def generate_audio(self, text, index):
        audio_path = f"{self.output_dir}/audio/audio_{index}.mp3"
        asyncio.run(self.generate_tts(text, audio_path))

        return audio_path

    async def generate_tts(self, text, output_path='short_audio.mp3', voice='en-US-GuyNeural', style='cheerful', rate='+0'):
        communicate_kwargs = {
            "text": text,
            "voice": voice,
            "rate": f"{rate}%"
        }

        # List of voices and their supported styles
        voice_data = await edge_tts.list_voices()
        selected_voice = next((v for v in voice_data if v["ShortName"] == voice), None)
        supported_styles = selected_voice.get("StyleList", []) if selected_voice else []

        # Apply style if supported
        if style and style in supported_styles:
            communicate_kwargs["style"] = style
            communicate_kwargs["styledegree"] = "2.0"  # Maximum expressiveness

        communicate = edge_tts.Communicate(**communicate_kwargs)
        await communicate.save(output_path)

    def get_audio_duration(self, audio_path):
        """Get audio length in seconds using ffprobe"""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)

    def make_clip(self, image_path, audio_path, duration, index):
        clip_path = f"{self.output_dir}/video/clip_{index}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}",
            clip_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return clip_path

    def concatenate_clips(self, clip_paths):
        list_path = f"{self.output_dir}/clips_list.txt"
        with open(list_path, 'w') as f:
            for path in clip_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")

        final_path = os.path.join(self.output_dir, "final_short_video1.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy", final_path
        ]
        subprocess.run(cmd)
        print(f"✅ Final video saved to: {final_path}")

    def run(self):
        clip_paths = []
        for idx, segment in enumerate(self.prompts):
            prompt = segment["prompt"]
            text = segment["text"]
            print(f"🎬 Generating clip for: '{prompt}'")

            image = self.generate_image(prompt, idx)
            audio = self.generate_audio(text, idx)
            duration = self.get_audio_duration(audio)

            clip = self.make_clip(image, audio, duration, idx)
            clip_paths.append(clip)

        self.concatenate_clips(clip_paths)

if __name__ == "__main__":
    creator = TextToVideoCreator("input.json")
    creator.run()
