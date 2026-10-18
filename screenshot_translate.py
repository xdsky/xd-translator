"""截图翻译模块：全局快捷键 → 框选截图 → OCR → 翻译 → 原位置覆盖显示。"""

import numpy as np
import mss
import keyboard
from PyQt5.QtCore import Qt, QRect, QObject, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget

import config


# ===================== 区域选择窗口 =====================
class RegionSelector(QWidget):
    """全屏半透明窗口，拖拽选择截图区域。ESC 取消。"""

    def __init__(self):
        super().__init__()
        self.start_pos = None
        self.end_pos = None
        self.selected = False
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.setCursor(Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(0, 120, 255), 2))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos:
            self.end_pos = event.pos()
            self.selected = True
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.selected = False
            self.close()

    def get_region(self):
        if not self.selected or not self.start_pos or not self.end_pos:
            return None
        rect = QRect(self.start_pos, self.end_pos).normalized()
        if rect.width() < 5 or rect.height() < 5:
            return None
        return rect


# ===================== 翻译结果覆盖窗口 =====================
class TranslationOverlay(QWidget):
    """在屏幕原文字位置显示翻译结果的透明窗口。点击/ESC 关闭。"""

    def __init__(self, results):
        super().__init__()
        self.results = results
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Microsoft YaHei", config.SCREENSHOT_FONT_SIZE)
        painter.setFont(font)

        for item in self.results:
            x, y, w, h = item["bbox"]
            text = item["text"]
            bg = QColor(*config.SCREENSHOT_BG_COLOR)
            painter.fillRect(x, y, w, h, bg)
            painter.setPen(QColor(*config.SCREENSHOT_TEXT_COLOR))
            painter.drawText(
                QRect(x, y, w, h),
                Qt.AlignCenter | Qt.TextWordWrap,
                text,
            )

    def mousePressEvent(self, event):
        self.close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Space):
            self.close()


# ===================== 截图翻译主逻辑 =====================
class ScreenshotTranslator(QObject):
    triggered = pyqtSignal()  # 用于跨线程触发 run()

    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self._ocr_cache = {}  # key: "ja"/"default" -> RapidOCR 实例
        self._overlay = None
        self.triggered.connect(self.run)

    def _get_ocr(self, src_lang):
        """按源语言选择 OCR 模型：日文用 PP-OCRv4 日文模型，其他用自带中英模型。"""
        key = "ja" if src_lang == "ja" else "default"
        if key not in self._ocr_cache:
            from rapidocr_onnxruntime import RapidOCR
            if key == "ja":
                self._ocr_cache[key] = self._build_ja_ocr()
            else:
                print("[OCR] 初始化 RapidOCR（中英模型）...", flush=True)
                self._ocr_cache[key] = RapidOCR()
        return self._ocr_cache[key]

    def _build_ja_ocr(self):
        """构建日文 OCR（PP-OCRv4 日文识别模型），失败则回退中英模型。"""
        from rapidocr_onnxruntime import RapidOCR
        try:
            import os
            from huggingface_hub import hf_hub_download
            os.makedirs(config.OCR_MODEL_DIR, exist_ok=True)
            print("[OCR] 准备日文识别模型...", flush=True)
            onnx = hf_hub_download(
                config.OCR_JA_REPO, "model.onnx", local_dir=config.OCR_MODEL_DIR
            )
            dct = hf_hub_download(
                config.OCR_JA_REPO, "japan_dict.txt", local_dir=config.OCR_MODEL_DIR
            )
            print("[OCR] 初始化 RapidOCR（日文模型）...", flush=True)
            return RapidOCR(rec_model_path=str(onnx), rec_keys_path=str(dct))
        except Exception as e:
            print(f"[OCR] 日文模型加载失败，回退中英模型: {e}", flush=True)
            return RapidOCR()

    def capture_region(self):
        """全屏截图并让用户框选区域，返回 (image_np, offset)。"""
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            img = np.array(sct.grab(monitor))[:, :, :3]  # BGRA -> BGR
            img_rgb = img[:, :, ::-1].copy()

        selector = RegionSelector()
        selector.show()
        app = QApplication.instance()
        while selector.isVisible():
            app.processEvents()

        rect = selector.get_region()
        if rect is None:
            return None, None

        x, y = rect.x(), rect.y()
        w, h = rect.width(), rect.height()
        region = img_rgb[y:y + h, x:x + w]
        return region, (x, y, w, h)

    def run(self):
        print("[截图翻译] 开始...", flush=True)
        region, offset = self.capture_region()
        if region is None:
            print("[截图翻译] 已取消", flush=True)
            return

        ox, oy, ow, oh = offset
        # OCR 源语言：用配置的 SOURCE_LANG，无则默认日语（常见场景）
        src_lang = config.SOURCE_LANG or "ja"
        print(f"[截图翻译] 区域: {offset}, OCR 识别中（源语言={src_lang}）...", flush=True)

        ocr = self._get_ocr(src_lang)
        result, _ = ocr(region)
        if not result:
            print("[截图翻译] 未识别到文字", flush=True)
            return

        items = []
        for box, text, conf in result:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x = int(min(xs)) + ox
            y = int(min(ys)) + oy
            w = int(max(xs) - min(xs))
            h = int(max(ys) - min(ys))
            if not text.strip():
                continue
            items.append({"bbox": (x, y, w, h), "text": text.strip()})

        print(f"[截图翻译] 识别到 {len(items)} 行，翻译中...", flush=True)

        translated = []
        texts = [item["text"] for item in items]
        try:
            translations = self.translator.batch_translate(texts, src_lang)
        except Exception as e:
            print(f"[截图翻译] 批量翻译失败: {e}", flush=True)
            translations = texts

        for item, trans in zip(items, translations):
            translated.append({"bbox": item["bbox"], "text": trans})

        if self._overlay is not None:
            self._overlay.close()
        self._overlay = TranslationOverlay(translated)
        self._overlay.show()
        print("[截图翻译] 完成（点击/ESC 关闭覆盖层）", flush=True)

    def register_hotkey(self, hotkey=None):
        key = hotkey or config.SCREENSHOT_HOTKEY
        keyboard.add_hotkey(key, lambda: self.triggered.emit())
        print(f"[截图翻译] 快捷键已注册: {key}", flush=True)
