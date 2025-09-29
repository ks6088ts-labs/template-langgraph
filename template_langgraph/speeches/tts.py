import io
import logging

from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup

from template_langgraph.loggers import get_logger

logger = get_logger(
    name=__name__,
    verbosity=logging.DEBUG,
)


class TtsWrapper:
    def __init__(self):
        pass

    def load_model(self):
        pass

    def synthesize_audio(
        self,
        text: str,
        language: str = "ja",
        speed: float = 1.0,
        pitch_shift: int = 0,
        volume_db: float = 0.0,
    ) -> bytes | None:
        """Convert text to speech audio using gTTS and pydub adjustments."""

        if not text.strip():
            return None

        try:
            tts = gTTS(text=text, lang=language)
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)

            audio_segment = AudioSegment.from_file(mp3_buffer, format="mp3")
            original_rate = audio_segment.frame_rate

            if pitch_shift != 0:
                semitone_ratio = 2.0 ** (pitch_shift / 12.0)
                shifted = audio_segment._spawn(
                    audio_segment.raw_data,
                    overrides={"frame_rate": int(original_rate * semitone_ratio)},
                )
                audio_segment = shifted.set_frame_rate(original_rate)

            if speed != 1.0:
                if speed > 1.0:
                    audio_segment = speedup(audio_segment, playback_speed=float(speed))
                else:
                    slowed_rate = max(int(original_rate * float(speed)), 1)
                    audio_segment = audio_segment._spawn(
                        audio_segment.raw_data,
                        overrides={"frame_rate": slowed_rate},
                    ).set_frame_rate(original_rate)

            if volume_db != 0:
                audio_segment += float(volume_db)

            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format="mp3")
            return output_buffer.getvalue()
        except Exception as e:  # pragma: no cover
            logger.error(f"Error in synthesize_audio: {e}")
            return None
