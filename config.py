import os

# 本地私有配置（含密钥，已在 .gitignore 中，不会被提交）
try:
    from config_local import LLM_API_KEY as _LOCAL_KEY
except ImportError:
    _LOCAL_KEY = None

# ===================== 音频采集 =====================
SAMPLE_RATE_OUT = 16000      # 送给 VAD/ASR 的采样率
BLOCK_MS = 10                # 采集块时长（毫秒）

# ===================== VAD =====================
HANGOVER_MS = 250            # 静音多久算一句话结束，调小降低延迟
MAX_SEGMENT_S = 6            # 单段最大时长（秒），调小让翻译更及时
MIN_SEGMENT_S = 0.3          # 单段最小时长（秒），过滤噪声
VAD_THRESHOLD = 0.5          # Silero VAD 语音概率阈值

# ===================== ASR =====================
ASR_MODEL_SIZE = "large-v3"  # faster-whisper 模型: tiny/base/small/medium/large-v3
ASR_DEVICE = "cuda"          # cuda 或 cpu
ASR_COMPUTE_TYPE = "float16"  # float16 / int8 / int8_float16
ASR_BEAM_SIZE = 3            # 1=greedy最快，3-5=beam精度更高（3是速度/精度平衡点）
LANG_CACHE_THRESHOLD = 0.8   # 语种检测置信度低于此值时重新检测
SOURCE_LANG = None           # 源语言（Whisper 代码），None=自动检测；指定后跳过自动检测更快

# 源语言选项（None 表示自动检测）
SOURCE_LANG_OPTIONS = [
    (None, "自动检测"),
    ("zh", "中文"),
    ("en", "English"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("ar", "العربية"),
    ("hi", "हिन्दी"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("th", "ไทย"),
    ("vi", "Tiếng Việt"),
    ("id", "Bahasa Indonesia"),
]

# ASR 模型选项（用于下拉菜单）
ASR_MODEL_OPTIONS = [
    ("tiny", "tiny (最快，精度低)"),
    ("base", "base (较快)"),
    ("small", "small (均衡)"),
    ("medium", "medium (推荐)"),
    ("large-v3", "large-v3 (最精确，慢)"),
]

# ===================== 翻译 =====================
TRANSLATION_ENGINE = "llm"      # 翻译引擎: "local"(本地模型) 或 "llm"(大模型API)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATION_DEVICE = "cuda"  # cuda 或 cpu
TARGET_LANG = "zh"           # 默认目标语言（Whisper 代码），可改 en/ja/ko 等
FALLBACK_LANG = "en"         # 当源语言 == 目标语言时，翻译成此语言
TRANSLATION_BEAM_SIZE = 4    # 1=greedy最快，4=beam精度更高
TRANSLATION_MAX_TOKENS = 256  # 单句最大生成 token 数，避免过长拖慢速度

# LLM 大模型翻译配置（OpenAI 兼容接口，支持 DeepSeek/Qwen/Kimi/OpenAI 等）
LLM_API_URL = "https://api.moonshot.cn/v1/chat/completions"  # Kimi API 地址
LLM_API_KEY = _LOCAL_KEY or os.environ.get("LLM_API_KEY", "")               # API Key
LLM_MODEL = "kimi-k2.6"        # 模型: kimi-k2.6 / kimi-k2.7-code
LLM_TEMPERATURE = 1            # 思考模式下的温度（Kimi 思考模式固定 1.0）
LLM_TIMEOUT = 60               # 请求超时（秒）
LLM_DISABLE_THINKING = True    # 关闭推理模型的思考过程（Kimi k2.6），提速 3-5 倍

# 翻译引擎选项
TRANSLATION_ENGINE_OPTIONS = [
    ("llm", "大模型 API（推荐）"),
    ("local", "本地模型（离线）"),
]

# 可选目标语言列表（Whisper 代码: 显示名称）
TARGET_LANG_OPTIONS = [
    ("zh", "简体中文"),
    ("zh-TW", "繁體中文"),
    ("en", "English"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("ar", "العربية"),
    ("hi", "हिन्दी"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("th", "ไทย"),
    ("vi", "Tiếng Việt"),
    ("id", "Bahasa Indonesia"),
]

# ===================== UI =====================
OVERLAY_WIDTH = 720
OVERLAY_HEIGHT = 180
OVERLAY_FONT_SIZE = 13

# ===================== 截图翻译 =====================
SCREENSHOT_HOTKEY = "ctrl+alt+t"     # 截图翻译快捷键
OCR_USE_GPU = False                   # OCR 是否用 GPU（需安装 onnxruntime-gpu）
OCR_MODEL_DIR = "models"              # OCR 模型存放目录
# 日文识别模型（RapidOCR 自带中文模型，日文识别需单独下载 PP-OCRv4 日文模型）
OCR_JA_REPO = "cycloneboy/japan_PP-OCRv4_rec_infer"
SCREENSHOT_FONT_SIZE = 14             # 翻译覆盖文字字号
SCREENSHOT_BG_COLOR = (0, 0, 0, 200)  # 覆盖背景色 RGBA（半透明黑）
SCREENSHOT_TEXT_COLOR = (255, 255, 255)  # 覆盖文字颜色 RGB
