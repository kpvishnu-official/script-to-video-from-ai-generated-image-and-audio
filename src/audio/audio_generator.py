import asyncio
import logging
import subprocess

import edge_tts


class AudioGenerator:
    def __init__(
        self,
        voice: str = "en-US-GuyNeural",
        rate: str = "+0%"
    ) -> None:
        self.voice = voice
        self.rate = rate

    def generate(self, text: str, output_path: str) -> str:
        """Generate TTS audio from text and save to output_path. Returns the path."""
        asyncio.run(self._generate_tts(text, output_path))
        logging.info(f"🔊 Audio saved: {output_path}")
        return output_path

    async def _generate_tts(self, text: str, output_path: str) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate
        )
        await communicate.save(output_path)

    def get_duration(self, audio_path: str) -> float:
        """Return the duration of an audio file in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logging.warning(f"⚠️ ffprobe failed: {e}")
            return 5.0
