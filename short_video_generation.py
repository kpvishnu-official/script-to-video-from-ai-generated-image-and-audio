import os
import json
import gc
import logging
import random
import argparse
import subprocess
from typing import List, Dict, Tuple

import torch

from src.image.image_generator import ImageGenerator
from src.audio.audio_generator import AudioGenerator


# ---------------- TRANSITIONS ---------------- #
TRANSITIONS = [
    "fade",
    "fadeblack",
    "fadewhite",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
    "radial",
    "zoomin",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "dissolve",
    "squeezeh",
    "squeezev",
]

TRANSITION_DURATION = 0.5  # seconds


# ---------------- CLI ---------------- #
def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="info")
    return parser.parse_args()


# ---------------- MAIN CLASS ---------------- #
class TextToVideoCreator:
    def __init__(
        self,
        json_path: str,
        output_dir: str = "media",
        resolution: Tuple[int, int] = (512, 768),
        transition_duration: float = TRANSITION_DURATION,
    ) -> None:

        self.json_path: str = json_path
        self.output_dir: str = output_dir
        self.resolution: Tuple[int, int] = resolution
        self.transition_duration: float = transition_duration

        self.prompts: List[Dict] = self.load_json()

        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/video", exist_ok=True)

        self.image_generator = ImageGenerator(
            model_name="Meina/MeinaMix_V11",
            device="cpu",
            resolution=self.resolution,
            use_large_model=True
        )

        self.audio_generator = AudioGenerator(
            voice="en-US-GuyNeural",
            rate="+0%"
        )

    # ---------------- LOAD ---------------- #
    def load_json(self) -> List[Dict]:
        with open(self.json_path, 'r') as f:
            return json.load(f)

    # ---------------- VIDEO ---------------- #
    def make_clip(self, image_path: str, audio_path: str, duration: float, index: int) -> str:
        clip_path: str = f"{self.output_dir}/video/clip_{index}.mp4"

        subprocess.run([
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}",
            clip_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return clip_path

    # ---------------- TRANSITIONS ---------------- #
    def concatenate_clips_with_transitions(
        self,
        clip_paths: List[str],
        durations: List[float]
    ) -> None:
        """
        Joins clips using FFmpeg xfade filters with a randomly chosen transition
        between each pair of clips. Audio streams are crossfaded with acrossfade
        to match. All re-encoding happens in a single FFmpeg pass.
        """
        final_path: str = os.path.join(self.output_dir, "final_video.mp4")

        if len(clip_paths) == 1:
            subprocess.run([
                "ffmpeg", "-y", "-i", clip_paths[0], "-c", "copy", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"✅ Final video saved: {final_path}")
            return

        td = self.transition_duration
        n = len(clip_paths)

        # Build -i inputs
        inputs: List[str] = []
        for path in clip_paths:
            inputs += ["-i", path]

        # Chain xfade (video) and acrossfade (audio) filters.
        # The xfade offset is the cumulative duration of clips seen so far,
        # minus the overlap introduced by each transition.
        filter_parts: List[str] = []
        v_prev = "[0:v]"
        a_prev = "[0:a]"
        offset = durations[0] - td
        chosen_transitions: List[str] = []

        for i in range(1, n):
            transition = random.choice(TRANSITIONS)
            chosen_transitions.append(transition)

            v_out = f"[vx{i}]"
            a_out = f"[ax{i}]"

            filter_parts.append(
                f"{v_prev}[{i}:v]xfade="
                f"transition={transition}:"
                f"duration={td}:"
                f"offset={offset:.4f}"
                f"{v_out}"
            )
            filter_parts.append(
                f"{a_prev}[{i}:a]acrossfade=d={td}{a_out}"
            )

            v_prev = v_out
            a_prev = a_out

            logging.info(
                f"🎞️  Transition {i}/{n - 1}: '{transition}' at offset {offset:.2f}s"
            )

            if i < n - 1:
                offset += durations[i] - td

        filter_complex = "; ".join(filter_parts)

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex", filter_complex,
                "-map", v_prev,
                "-map", a_prev,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                final_path,
            ]
        )

        logging.info("🔗 Concatenating clips with transitions...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"✅ Final video saved: {final_path}")
        logging.info(f"🎲 Transitions used: {', '.join(chosen_transitions)}")

    # ---------------- RUN ---------------- #
    def run(self) -> None:
        clip_paths: List[str] = []
        clip_durations: List[float] = []
        updated_scenes: List[Dict] = []

        for idx, segment in enumerate(self.prompts):
            prompt: str = segment["prompt"]
            text: str = segment["text"]

            logging.info(f"🎬 Processing scene {idx}")

            image_path = f"{self.output_dir}/images/frame_{idx}.png"
            audio_path = f"{self.output_dir}/audio/audio_{idx}.mp3"

            image, final_prompt = self.image_generator.generate(
                prompt=prompt,
                output_path=image_path
            )

            audio = self.audio_generator.generate(text, audio_path)
            duration: float = self.audio_generator.get_duration(audio)

            clip = self.make_clip(image, audio, duration, idx)
            clip_paths.append(clip)
            clip_durations.append(duration)

            updated_scenes.append({
                "start": segment["start"],
                "end": segment["end"],
                "prompt": final_prompt,
                "text": text,
                "image": image,
                "audio": audio,
                "clip": clip,
                "duration": duration,
            })

        self.concatenate_clips_with_transitions(clip_paths, clip_durations)

        output_json = os.path.join(self.output_dir, "updated_scenes.json")
        with open(output_json, "w") as f:
            json.dump(updated_scenes, f, indent=2)

        logging.info(f"🧾 Saved: {output_json}")

        self.image_generator.cleanup()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------- ENTRY ---------------- #
if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.log)

    creator = TextToVideoCreator("input.json")
    creator.run()
