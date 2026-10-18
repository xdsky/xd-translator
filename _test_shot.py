"""端到端测试截图翻译：生成真实图片 -> OCR -> 批量翻译，打印每一阶段。"""
import os, sys, glob, json
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
_nvidia_base = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia")
for _d in glob.glob(os.path.join(_nvidia_base, "*", "bin")):
    os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
config.TARGET_LANG = "zh"

# 生成测试图片：模拟日语字幕/网页
lines = [
    "北海道新聞の記者がスパイ容疑で拘束された",
    "首都モスクワにある日本大使館は声明を発表した",
    "大臣を続けるのは難しいと述べた",
    "なっとくのいく説明ができないのであれば",
    "この問題について詳しく説明してください",
]
img = Image.new("RGB", (900, 320), (255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
for i, t in enumerate(lines):
    draw.text((20, 20 + i * 58), t, fill=(0, 0, 0), font=font)
img.save("_test_shot.png")
region = np.array(img)  # RGB
print(f"[1] 测试图已生成，{len(lines)} 行")
print()

# OCR
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
result, _ = ocr(region)
print(f"[2] OCR 识别到 {len(result) if result else 0} 个文本框")
ocr_texts = []
for box, text, conf in (result or []):
    ocr_texts.append(text.strip())
    print(f"    conf={conf:.2f}  {text!r}")
print()

# 翻译
from translator import Translator
t = Translator()
print(f"[3] 批量翻译 (src=ja, engine={config.TRANSLATION_ENGINE})")
translations = t.batch_translate(ocr_texts, "ja")
print()
for o, r in zip(ocr_texts, translations):
    print(f"    原文: {o!r}")
    print(f"    译文: {r!r}")
    print()
