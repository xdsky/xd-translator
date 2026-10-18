# 自动翻译（Audio Translator）

实时音频翻译 + 截图翻译的小工具，基于 PyQt5 的桌面悬浮窗。

## 功能

- **实时音频翻译**：捕获系统正在播放的声音 → 语音识别 → 自动翻译 → 悬浮窗显示原文与译文
- **截图翻译**：按 `Ctrl+Alt+T` 框选屏幕区域 → OCR 识别 → 翻译 → 原文位置覆盖显示
- **可切换设置**：源语言（含自动检测）、目标语言、音频设备、ASR 模型、翻译引擎

## 工作原理

```
实时音频：系统回环音频 → VAD 分段 → 语音识别 → 翻译 → 悬浮窗
截图翻译：快捷键 → 框选截图 → OCR → 翻译 → 覆盖显示
```

| 环节 | 用到的技术 |
|---|---|
| 音频采集 | `soundcard`（系统回环 loopback） |
| 语音分段 | `silero-vad` |
| 语音识别 | `faster-whisper`（含语种自动检测） |
| 翻译 | 本地 NLLB 模型 或 大模型 API（OpenAI 兼容接口） |
| OCR | `rapidocr_onnxruntime` |
| 界面 | `PyQt5` |

## 环境要求

- Windows
- Python 3.10+
- NVIDIA GPU（可选）：有 GPU 时用 CUDA 加速，纯 CPU 也能跑但更慢

## 安装

```bash
git clone https://github.com/xdsky/xd-translator.git
cd xd-translator
python -m venv .venv
.venv\Scripts\activate

# 1) 深度学习框架（CUDA 版，按需选择版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) 其余依赖（含截图翻译所需）
pip install -r requirements.txt
```

## 配置

配置文件是 [config.py](config.py)，里面集中了所有参数（采样率、VAD 阈值、模型大小、目标语言、快捷键等）。

**API Key 的存放方式**（不要把密钥写进源码或提交到仓库）：

- 方式一（推荐）：在项目根目录新建 `config_local.py`，内容为

  ```python
  LLM_API_KEY = "你的 API Key"
  ```

  该文件已在 `.gitignore` 中，不会被提交。

- 方式二：不建文件，改为设置环境变量 `LLM_API_KEY`。

> 未配置 API Key 时，可将 `config.TRANSLATION_ENGINE` 设为 `local`，改用本地 NLLB 模型离线翻译。

## 运行

```bash
python main.py
```

首次运行会自动下载模型（默认使用 `hf-mirror.com` 镜像加速）。

## 使用

- 悬浮窗实时显示「识别原文 + 译文」
- 设置面板可即时切换：目标语言 / 源语言 / 音频设备 / ASR 模型 / 翻译引擎
- 截图翻译快捷键：`Ctrl+Alt+T`

## 目录结构

| 文件 | 作用 |
|---|---|
| `main.py` | 主程序，编排各模块与线程 |
| `config.py` | 全部配置项 |
| `audio_capture.py` | 系统回环音频采集 |
| `vad.py` | 语音活动检测与分段 |
| `asr.py` | faster-whisper 语音识别 + 语种检测 |
| `translator.py` | 翻译（本地 NLLB 模型 / 大模型 API） |
| `overlay.py` | 悬浮窗界面 |
| `screenshot_translate.py` | 截图翻译（OCR + 覆盖显示） |
| `models/` | OCR 模型文件（含日文识别模型） |
| `_test_shot.py` | 截图 OCR 的独立调试脚本 |

## 许可

本项目暂未指定开源许可证。如需开源，建议在仓库根目录添加 `LICENSE` 文件（如 MIT、Apache-2.0）。

> 练习说明：本节由 `add-license-section` 分支添加，用于演示「分支 → PR → 合并」流程。

