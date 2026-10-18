"""音频采集模块：通过 WASAPI loopback 捕获系统扬声器输出。"""

import numpy as np
import soundcard as sc

import config


def list_loopback_devices():
    """返回可用的 loopback 麦克风列表 [(name, id), ...]"""
    devices = []
    for m in sc.all_microphones(include_loopback=True):
        if getattr(m, "isloopback", False):
            devices.append((m.name, m.id))
    # 如果没有标记为 loopback 的，返回所有设备供选择
    if not devices:
        devices = [(m.name, m.id) for m in sc.all_microphones(include_loopback=True)]
    return devices


class AudioCapture:
    def __init__(self, device_id=None, native_sr=48000):
        self.native_sr = native_sr  # Windows WASAPI 常用 48000Hz
        if device_id:
            # 按 id 查找设备
            all_mics = sc.all_microphones(include_loopback=True)
            found = [m for m in all_mics if m.id == device_id]
            self.mic = found[0] if found else all_mics[0]
        else:
            # 找到默认扬声器对应的 loopback 麦克风
            speaker_name = sc.default_speaker().name
            loopback_mics = [
                m for m in sc.all_microphones(include_loopback=True)
                if getattr(m, "isloopback", False)
            ]
            if not loopback_mics:
                loopback_mics = [
                    m for m in sc.all_microphones(include_loopback=True)
                    if m.name == speaker_name
                ]
            if not loopback_mics:
                loopback_mics = sc.all_microphones(include_loopback=True)
            self.mic = loopback_mics[0]

    def start(self, audio_queue, stop_event):
        block = int(self.native_sr * config.BLOCK_MS / 1000)
        try:
            ctx = self.mic.recorder(
                samplerate=self.native_sr, channels=2, blocksize=block
            )
        except Exception:
            ctx = self.mic.recorder(channels=2, blocksize=block)
        with ctx as rec:
            while not stop_event.is_set():
                try:
                    data = rec.record(numframes=block)  # (block, 2) float32
                except Exception:
                    continue
                mono = data.mean(axis=1)
                if self.native_sr != config.SAMPLE_RATE_OUT:
                    mono = self._resample(mono, self.native_sr, config.SAMPLE_RATE_OUT)
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except Exception:
                        pass
                audio_queue.put(mono.astype(np.float32))

    @staticmethod
    def _resample(x, sr_in, sr_out):
        duration = len(x) / sr_in
        n_out = int(duration * sr_out)
        idx = np.linspace(0, len(x) - 1, n_out)
        return np.interp(idx, np.arange(len(x)), x)
