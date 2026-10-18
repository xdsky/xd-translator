"""主程序：编排音频采集、VAD、ASR、翻译、UI 显示。"""

import os
import sys

# Windows 控制台强制 UTF-8，避免中文 print 乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 国内 HuggingFace 镜像（需在导入 transformers/faster_whisper 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 将 nvidia CUDA 运行库（cublas/cudnn 等）加入 PATH，供 ctranslate2 加载
import glob
_nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
for _d in glob.glob(os.path.join(_nvidia_base, "*", "bin")):
    if _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import queue
import signal
import threading

import config
from audio_capture import AudioCapture, list_loopback_devices
from vad import Segmenter
from asr import ASR
from translator import Translator

# PyQt5 需在 torch/transformers 之后导入，避免 DLL 冲突
from PyQt5.QtWidgets import QApplication

from overlay import OverlayWindow


def main():
    print("[main] starting...", flush=True)
    app = QApplication([])
    win = OverlayWindow()
    win.show()
    print("[main] window shown", flush=True)

    q_audio = queue.Queue(maxsize=10)
    q_segment = queue.Queue(maxsize=4)
    q_text = queue.Queue(maxsize=4)
    stop_all = threading.Event()

    # 先加载所有模型
    print("[main] loading VAD...", flush=True)
    seg = Segmenter()
    print("[main] VAD loaded", flush=True)

    print("[main] loading ASR model...", flush=True)
    asr = ASR()
    print("[main] ASR model loaded", flush=True)

    print("[main] loading translator model...", flush=True)
    translator = Translator()
    print("[main] translator loaded", flush=True)

    # ========== 截图翻译 ==========
    from screenshot_translate import ScreenshotTranslator
    ss_trans = ScreenshotTranslator(translator)
    ss_trans.register_hotkey()

    # ========== 音频采集线程（可重启） ==========
    audio_stop = threading.Event()
    current_device_id = [None]

    def start_audio_thread(device_id=None):
        nonlocal audio_stop
        audio_stop.set()  # 停止旧线程
        audio_stop = threading.Event()
        cap = AudioCapture(device_id=device_id)
        current_device_id[0] = device_id or cap.mic.id
        threading.Thread(
            target=cap.start, args=(q_audio, audio_stop), daemon=True
        ).start()

    print("[main] init audio capture...", flush=True)
    start_audio_thread()
    print("[main] audio capture thread started", flush=True)

    # 填充音频设备下拉框
    devices = list_loopback_devices()
    win.set_audio_devices(devices, current=current_device_id[0])

    # ========== VAD 分段线程 ==========
    def vad_loop():
        while not stop_all.is_set():
            try:
                chunk = q_audio.get(timeout=0.5)
                seg.process(chunk, q_segment)
            except queue.Empty:
                continue

    threading.Thread(target=vad_loop, daemon=True).start()

    # ========== ASR 识别线程 ==========
    asr_lock = threading.Lock()  # 切换模型时加锁

    def asr_loop():
        while not stop_all.is_set():
            try:
                audio = q_segment.get(timeout=0.5)
                with asr_lock:
                    text, lang, prob = asr.transcribe(audio)
                if text:
                    if q_text.full():
                        try:
                            q_text.get_nowait()
                        except Exception:
                            pass
                    q_text.put((text, lang))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ASR 错误] {e}")

    threading.Thread(target=asr_loop, daemon=True).start()

    # ========== 翻译 -> UI 线程 ==========
    def trans_loop():
        while not stop_all.is_set():
            try:
                text, lang = q_text.get(timeout=0.5)
                try:
                    translated = translator.translate(text, lang)
                except Exception as e:
                    translated = f"[翻译失败] {e}"
                win.new_text.emit(text, translated)
            except queue.Empty:
                continue

    threading.Thread(target=trans_loop, daemon=True).start()

    # ========== 设置面板信号处理 ==========
    def on_target_lang_changed(code):
        config.TARGET_LANG = code
        print(f"[设置] 目标语言 -> {code}", flush=True)

    def on_source_lang_changed(code):
        config.SOURCE_LANG = code
        label = "自动检测" if code is None else code
        print(f"[设置] 源语言 -> {label}", flush=True)

    def on_audio_device_changed(dev_id):
        print(f"[设置] 音频设备 -> {dev_id}", flush=True)
        start_audio_thread(device_id=dev_id)

    def on_asr_model_changed(model_size):
        print(f"[设置] ASR 模型 -> {model_size}，重新加载...", flush=True)
        with asr_lock:
            nonlocal asr
            del asr.model
            asr = ASR(model_size=model_size)
        config.ASR_MODEL_SIZE = model_size
        print("[设置] ASR 模型加载完成", flush=True)

    win.target_lang_changed.connect(on_target_lang_changed)
    win.source_lang_changed.connect(on_source_lang_changed)
    win.audio_device_changed.connect(on_audio_device_changed)
    win.asr_model_changed.connect(on_asr_model_changed)
    win.engine_changed.connect(lambda e: setattr(config, "TRANSLATION_ENGINE", e) or print(f"[设置] 翻译引擎 -> {e}", flush=True))

    def handle_sigint(*_):
        stop_all.set()
        audio_stop.set()
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    print("[main] entering event loop...", flush=True)
    app.exec_()
    stop_all.set()
    audio_stop.set()
    print("[main] exited", flush=True)


if __name__ == "__main__":
    main()
