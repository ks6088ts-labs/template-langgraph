import logging

import whisper

from template_langgraph.loggers import get_logger

logger = get_logger(
    name=__name__,
    verbosity=logging.DEBUG,
)


class SttWrapper:
    def __init__(self):
        self.model = None

    def load_model(self, model_size: str):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)

    def transcribe(
        self,
        audio_path: str,
        language: str,
    ) -> str:
        logger.info(f"Transcribing audio: {audio_path} with language: {language}")
        result = self.model.transcribe(
            audio=audio_path,
            language=language,
        )
        return result["text"]
