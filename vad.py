"""VAD 模块：使用 Silero VAD 按停顿切分语音段。"""

import numpy as np
import torch
from silero_vad import load_silero_vad

import config


class Segmenter:
    def __init__(self):
        self.model = load_silero_vad()
        self.SR = config.SAMPLE_RATE_OUT
        self.VAD_CHUNK = 512  # Silero 要求: 512/1024/1536/3072
        self.reset()

    def reset(self):
        self.segment_buf = []       # 累积的完整语音段（用于输出）
        self.vad_remainder = np.array([], dtype=np.float32)  # VAD 未满 512 的残留
        self.speaking = False
        self.silence_frames = 0
        self.hangover_frames = int(
            config.HANGOVER_MS / (self.VAD_CHUNK / self.SR * 1000)
        )

    def process(self, audio_16k, out_queue):
        # 累积到完整段缓冲（保留所有音频，flush 时输出）
        self.segment_buf.append(audio_16k)

        # VAD 处理：拼接残留 + 新音频
        vad_input = np.concatenate([self.vad_remainder, audio_16k])
        offset = 0
        while len(vad_input) - offset >= self.VAD_CHUNK:
            chunk = vad_input[offset : offset + self.VAD_CHUNK]
            offset += self.VAD_CHUNK
            prob = self.model(torch.from_numpy(chunk), self.SR).item()
            is_speech = prob > config.VAD_THRESHOLD
            if is_speech:
                self.speaking = True
                self.silence_frames = 0
            elif self.speaking:
                self.silence_frames += 1
                if self.silence_frames >= self.hangover_frames:
                    self._flush(out_queue)
        self.vad_remainder = vad_input[offset:]

        # 最大段长强制刷新
        total = sum(len(b) for b in self.segment_buf)
        if total >= config.MAX_SEGMENT_S * self.SR:
            self._flush(out_queue)

    def _flush(self, out_queue):
        if not self.segment_buf:
            return
        seg = np.concatenate(self.segment_buf)
        if len(seg) >= config.MIN_SEGMENT_S * self.SR:
            if out_queue.full():
                try:
                    out_queue.get_nowait()
                except Exception:
                    pass
            out_queue.put(seg.astype(np.float32))
        self.reset()
