"""悬浮显示窗：无边框、置顶、半透明、可拖拽，带翻译历史和设置面板。"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

import config

MAX_HISTORY = 20  # 保留最近 20 条翻译记录


class OverlayWindow(QWidget):
    new_text = pyqtSignal(str, str)            # 原文, 译文
    target_lang_changed = pyqtSignal(str)       # 新目标语言代码
    source_lang_changed = pyqtSignal(str)       # 新源语言代码（None=自动检测）
    audio_device_changed = pyqtSignal(str)      # 新设备名
    asr_model_changed = pyqtSignal(str)         # 新模型名
    engine_changed = pyqtSignal(str)            # 新翻译引擎

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._normal_height = 260  # 历史记录模式需要更高窗口
        self._settings_height = 250  # 源+目标语言+引擎+设备+模型 五行
        self.resize(config.OVERLAY_WIDTH, self._normal_height)

        # 深色半透明面板
        self.panel = QFrame(self)
        self.panel.setStyleSheet(
            "QFrame { background: rgba(0, 0, 0, 185); border-radius: 12px; }"
        )

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(14, 8, 14, 10)
        layout.setSpacing(4)

        # 顶栏：设置按钮 + 清空按钮 + 关闭按钮
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(24, 24)
        self.settings_btn.setStyleSheet(
            "QPushButton { color: #AAAAAA; background: transparent; border: none; font-size: 14px; }"
            "QPushButton:hover { color: #4FC3F7; }"
        )
        self.settings_btn.clicked.connect(self.toggle_settings)

        self.clear_btn = QPushButton("🗑")
        self.clear_btn.setFixedSize(24, 24)
        self.clear_btn.setStyleSheet(
            "QPushButton { color: #AAAAAA; background: transparent; border: none; font-size: 13px; }"
            "QPushButton:hover { color: #FF8A65; }"
        )
        self.clear_btn.setToolTip("清空历史记录")
        self.clear_btn.clicked.connect(self.clear_history)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet(
            "QPushButton { color: #AAAAAA; background: transparent; border: none; font-size: 14px; }"
            "QPushButton:hover { color: #FFFFFF; }"
        )
        self.close_btn.clicked.connect(self.close)

        top_bar.addWidget(self.settings_btn)
        top_bar.addWidget(self.clear_btn)
        top_bar.addStretch()
        top_bar.addWidget(self.close_btn)
        layout.addLayout(top_bar)

        # 翻译历史滚动区
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFrameShape(QFrame.NoFrame)
        self.history_scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QScrollBar:vertical { background: transparent; width: 6px; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,80); border-radius: 3px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        self.history_container = QWidget()
        self.history_container.setStyleSheet("QWidget { background: transparent; }")
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(2, 2, 2, 2)
        self.history_layout.setSpacing(8)
        self.history_layout.addStretch()
        self.history_scroll.setWidget(self.history_container)

        self.history_scroll.setMinimumHeight(140)
        layout.addWidget(self.history_scroll, stretch=1)

        # 设置面板
        self.settings_frame = QFrame()
        self.settings_frame.setStyleSheet(
            "QFrame { background: rgba(255,255,255,15); border-radius: 8px; }"
            "QLabel { color: #CCCCCC; font-size: 12px; }"
            "QComboBox { background: rgba(255,255,255,30); color: #FFFFFF; "
            "border: 1px solid #555; border-radius: 4px; padding: 3px 8px; min-width: 120px; }"
            "QComboBox:hover { border-color: #4FC3F7; }"
            "QComboBox QAbstractItemView { background: #2a2a2a; color: #FFFFFF; selection-background-color: #4FC3F7; }"
        )
        settings_layout = QVBoxLayout(self.settings_frame)
        settings_layout.setContentsMargins(10, 8, 10, 8)
        settings_layout.setSpacing(6)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("目标语言:"))
        self.lang_combo = QComboBox()
        for code, name in config.TARGET_LANG_OPTIONS:
            self.lang_combo.addItem(name, code)
        idx = self.lang_combo.findData(config.TARGET_LANG)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        settings_layout.addLayout(lang_row)

        # 源语言
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("源语言:"))
        self.source_combo = QComboBox()
        for code, name in config.SOURCE_LANG_OPTIONS:
            self.source_combo.addItem(name, code)
        idx = self.source_combo.findData(config.SOURCE_LANG)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self.source_combo)
        src_row.addStretch()
        settings_layout.addLayout(src_row)

        # 翻译引擎
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("翻译引擎:"))
        self.engine_combo = QComboBox()
        for code, name in config.TRANSLATION_ENGINE_OPTIONS:
            self.engine_combo.addItem(name, code)
        idx = self.engine_combo.findData(config.TRANSLATION_ENGINE)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        engine_row.addWidget(self.engine_combo)
        engine_row.addStretch()
        settings_layout.addLayout(engine_row)

        audio_row = QHBoxLayout()
        audio_row.addWidget(QLabel("音频设备:"))
        self.audio_combo = QComboBox()
        self.audio_combo.setMinimumWidth(180)
        self.audio_combo.currentIndexChanged.connect(self._on_audio_changed)
        audio_row.addWidget(self.audio_combo)
        audio_row.addStretch()
        settings_layout.addLayout(audio_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("识别模型:"))
        self.model_combo = QComboBox()
        for size, desc in config.ASR_MODEL_OPTIONS:
            self.model_combo.addItem(desc, size)
        idx = self.model_combo.findData(config.ASR_MODEL_SIZE)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        settings_layout.addLayout(model_row)

        self.settings_frame.setVisible(False)
        layout.addWidget(self.settings_frame)

        # 外层滚动区
        self.scroll = QScrollArea(self)
        self.scroll.setWidget(self.panel)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)

        self.new_text.connect(self._update_text)
        self.drag_pos = None
        self._settings_visible = False
        self._entries = []  # 历史记录条目列表

    def set_audio_devices(self, devices, current=None):
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        for name, dev_id in devices:
            self.audio_combo.addItem(name, dev_id)
        if current:
            idx = self.audio_combo.findData(current)
            if idx >= 0:
                self.audio_combo.setCurrentIndex(idx)
        self.audio_combo.blockSignals(False)

    def toggle_settings(self):
        self._settings_visible = not self._settings_visible
        self.settings_frame.setVisible(self._settings_visible)
        if self._settings_visible:
            self.resize(self.width(), self._normal_height + self._settings_height)
        else:
            self.resize(self.width(), self._normal_height)

    def clear_history(self):
        for entry in self._entries:
            self.history_layout.removeWidget(entry)
            entry.deleteLater()
        self._entries.clear()

    def _on_lang_changed(self, idx):
        self.target_lang_changed.emit(self.lang_combo.itemData(idx))

    def _on_source_changed(self, idx):
        self.source_lang_changed.emit(self.source_combo.itemData(idx))

    def _on_audio_changed(self, idx):
        dev_id = self.audio_combo.itemData(idx)
        if dev_id:
            self.audio_device_changed.emit(dev_id)

    def _on_model_changed(self, idx):
        self.asr_model_changed.emit(self.model_combo.itemData(idx))

    def _on_engine_changed(self, idx):
        self.engine_changed.emit(self.engine_combo.itemData(idx))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.panel.setGeometry(0, 0, self.width(), self.height())
        self.scroll.setGeometry(0, 0, self.width(), self.height())

    def _update_text(self, src, dst):
        # 创建新条目
        entry = QFrame()
        entry.setStyleSheet(
            "QFrame { background: rgba(255,255,255,8); border-radius: 6px; }"
        )
        entry_layout = QVBoxLayout(entry)
        entry_layout.setContentsMargins(8, 6, 8, 6)
        entry_layout.setSpacing(2)

        src_label = QLabel(src)
        src_label.setWordWrap(True)
        src_label.setFont(QFont("Microsoft YaHei", config.OVERLAY_FONT_SIZE - 1))
        src_label.setStyleSheet("color: #AAAAAA;")

        dst_label = QLabel(dst)
        dst_label.setWordWrap(True)
        dst_label.setFont(QFont("Microsoft YaHei", config.OVERLAY_FONT_SIZE))
        dst_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")

        entry_layout.addWidget(src_label)
        entry_layout.addWidget(dst_label)

        # 插入到末尾（stretch 之前）
        self.history_layout.insertWidget(self.history_layout.count() - 1, entry)
        self._entries.append(entry)

        # 超过最大数量，删除最早的
        while len(self._entries) > MAX_HISTORY:
            old = self._entries.pop(0)
            self.history_layout.removeWidget(old)
            old.deleteLater()

        # 自动滚到底部
        sb = self.history_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self.drag_pos and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self.drag_pos)
