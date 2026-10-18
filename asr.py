"""ASR 模块：使用 faster-whisper 进行语音识别与语种检测。"""

from faster_whisper import WhisperModel

import config


class ASR:
    def __init__(self, model_size=None, device=None):
        model_size = model_size or config.ASR_MODEL_SIZE
        device = device or config.ASR_DEVICE
        compute_type = config.ASR_COMPUTE_TYPE if device == "cuda" else "int8"
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.cached_lang = None

    def transcribe(self, audio_16k):
        language = config.SOURCE_LANG  # None=自动检测，指定则跳过检测直接识别
        segments, info = self.model.transcribe(
            audio_16k,
            language=language,
            beam_size=config.ASR_BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segments).strip()
        return text, info.language, info.language_probability
