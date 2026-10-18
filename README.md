# 自动翻译（Audio Translator）

[![CI](https://github.com/xdsky/xd-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/xdsky/xd-translator/actions/workflows/ci.yml)

**让系统里播放的外语，自动变成你的母语。**

一个 Windows 桌面小工具：**听**系统正在播放的声音 → **识别语种与内容** → **翻译** → **把译文浮在屏幕角落**。全程自动，不用复制粘贴、不用切窗口，看视频 / 开会 / 打游戏时都能用。

除实时音频翻译外，还支持**截图翻译**：按 `Ctrl+Alt+T` 框选屏幕上任意区域（游戏字幕、图片、PDF 均可），译文会覆盖在原文位置。

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

## 第三方模型与资源

本项目**自有代码**以 MIT 许可证开源；运行时会用到以下第三方模型与资源，其版权与许可归各自作者所有。

### 一、随仓库提供（`models/` 目录）

| 文件 | 来源 | 用途 |
|---|---|---|
| `model.onnx` | [cycloneboy/japan_PP-OCRv4_rec_infer](https://huggingface.co/cycloneboy/japan_PP-OCRv4_rec_infer) | PP-OCRv4 日文识别模型 |
| `japan_dict.txt` | 同上 | 日文识别字典 |

这两个文件由 `huggingface_hub` 从上述仓库下载（见 `config.py` 中的 `OCR_JA_REPO`），随本项目一起分发，**保留其原始许可**。

### 二、运行时自动下载

| 用途 | 来源 | 说明 |
|---|---|---|
| 中英 OCR | `rapidocr_onnxruntime` 自带模型 | 随安装包提供，无需额外下载 |
| 语音识别 | faster-whisper + Whisper 权重 | 首次运行从 HuggingFace 下载（默认走 `hf-mirror.com` 镜像） |
| 语音分段 | silero-vad | 首次运行时下载 |
| 本地翻译 | `facebook/nllb-200-distilled-600M` | 首次运行时下载 |

以上模型均可替换为其他模型。**能否商用请以其官方页面标注的许可为准**——例如 `facebook/nllb-200-distilled-600M` 在 HuggingFace 上标注的是非商用许可（CC-BY-NC-4.0），商用前请自行确认。

### 三、免责

本项目的 MIT 许可证**仅覆盖自有代码**，不覆盖上述第三方模型文件；这些资源仍受其原始许可约束。

## 许可

本项目采用 **MIT 许可证**开源，详见根目录的 [LICENSE](LICENSE) 文件——你可以自由使用、修改、分发，只需保留版权声明。

