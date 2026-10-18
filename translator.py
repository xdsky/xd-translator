"""翻译模块：支持 LLM 大模型 API 和本地模型两种引擎。

- LLM 引擎：调用大模型 API（DeepSeek/Qwen/Kimi/OpenAI 等），翻译质量最高
- 本地引擎：日→中/英→中 用 opus-mt 双语模型，其他用 NLLB-200
"""

import json
import time
import torch
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config

# Whisper 语言代码 -> 中文语言名（用于 LLM 翻译提示词）
LANG_NAMES = {
    "zh": "简体中文",
    "zh-TW": "繁體中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "hi": "印地语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "th": "泰语",
    "vi": "越南语",
    "id": "印尼语",
}

# Whisper 语言代码 -> NLLB-200 语言代码
WHISPER_TO_NLLB = {
    "zh": "zho_Hans",      # 简体中文
    "zh-TW": "zho_Hant",   # 繁体中文
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "ara_Arab",
    "hi": "hin_Deva",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "bg": "bul_Cyrl",
    "sk": "slk_Latn",
    "hr": "hrv_Latn",
    "lt": "lit_Latn",
    "lv": "lva_Latn",
    "et": "est_Latn",
    "sl": "slv_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "fa": "pes_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "ur": "urd_Arab",
    "ca": "cat_Latn",
    "sr": "srp_Cyrl",
}

# 专用双语模型（(源语言, 目标语言) -> 模型名）
# opus-mt 模型对特定语言对翻译质量远优于 NLLB
SPECIALIZED_MODELS = {
    ("ja", "zh"): "shun89/opus-mt-ja-zh",
    ("ja", "zh-TW"): "shun89/opus-mt-ja-zh",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "zh-TW"): "Helsinki-NLP/opus-mt-en-zh",
}


class Translator:
    # 类级冷却：上一次 429 的时间戳
    _last_429_time = 0
    _cooldown_seconds = 30

    def __init__(self, device=None):
        self.device = device or config.TRANSLATION_DEVICE
        # NLLB 通用模型（懒加载，仅本地引擎或 LLM 失败回退时才加载）
        self.nllb_tokenizer = None
        self.nllb_model = None
        # 专用模型缓存（按需懒加载）
        self._specialized = {}  # key: (src, tgt), value: (tokenizer, model)

    @classmethod
    def _in_cooldown(cls):
        return time.time() - cls._last_429_time < cls._cooldown_seconds

    def _load_nllb(self):
        if self.nllb_model is None:
            print(f"[翻译] 加载本地模型 {config.TRANSLATION_MODEL}...", flush=True)
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATION_MODEL)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_MODEL)
            self.nllb_model = self.nllb_model.to(self.device)
            if self.device == "cuda":
                self.nllb_model = self.nllb_model.half()
            self.nllb_model.eval()

    def _get_specialized(self, src_lang, tgt_lang):
        """获取专用模型，不存在则加载并缓存；加载失败返回 None 回退 NLLB。"""
        key = (src_lang, tgt_lang)
        if key not in self._specialized:
            model_name = SPECIALIZED_MODELS[key]
            try:
                print(f"[翻译] 加载专用模型 {model_name}...", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model = model.to(self.device)
                if self.device == "cuda":
                    model = model.half()
                model.eval()
                self._specialized[key] = (tokenizer, model)
            except Exception as e:
                print(f"[翻译] 专用模型加载失败，回退 NLLB: {e}", flush=True)
                self._specialized[key] = None
        return self._specialized[key]

    def _to_nllb(self, lang):
        return WHISPER_TO_NLLB.get(lang, "eng_Latn")

    def _translate_specialized(self, text, src_lang, tgt_lang):
        """用 opus-mt 双语模型翻译（不需要 src_lang 和 forced_bos_token_id）。"""
        result = self._get_specialized(src_lang, tgt_lang)
        if result is None:
            return self._translate_nllb(text, src_lang, tgt_lang)
        tokenizer, model = result
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                num_beams=config.TRANSLATION_BEAM_SIZE,
                max_new_tokens=config.TRANSLATION_MAX_TOKENS,
                early_stopping=True,
            )
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    def _translate_nllb(self, text, src_lang, tgt_lang):
        """用 NLLB-200 多语言模型翻译。"""
        self._load_nllb()
        src_nllb = self._to_nllb(src_lang)
        tgt_nllb = self._to_nllb(tgt_lang)

        self.nllb_tokenizer.src_lang = src_nllb
        tgt_token = self.nllb_tokenizer.convert_tokens_to_ids(tgt_nllb)

        inputs = self.nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            gen = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=tgt_token,
                num_beams=config.TRANSLATION_BEAM_SIZE,
                max_new_tokens=config.TRANSLATION_MAX_TOKENS,
                early_stopping=True,
            )
        return self.nllb_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    @staticmethod
    def _build_llm_body(messages):
        """构造 LLM 请求体。关闭思考模式时可提速 3-5 倍（Kimi k2.6）。
        非思考模式下 temperature 由服务端固定，不能显式传递。"""
        body = {"model": config.LLM_MODEL, "messages": messages}
        if getattr(config, "LLM_DISABLE_THINKING", False):
            body["thinking"] = {"type": "disabled"}
        else:
            body["temperature"] = config.LLM_TEMPERATURE
        return body

    def _translate_llm(self, text, src_lang, tgt_lang):
        """调用大模型 API 翻译，质量最高。"""
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

        system_prompt = (
            f"你是一位专业的翻译专家。请将用户提供的{src_name}文本翻译成{tgt_name}。"
            f"要求：1.只输出译文，不要添加任何解释、标点或前缀；"
            f"2.保持原文的语气和风格；3.专业术语准确；4.译文自然流畅。"
        )
        user_prompt = f"请翻译：{text}"

        try:
            resp = requests.post(
                config.LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {config.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=self._build_llm_body([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]),
                timeout=config.LLM_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.content
            # 尝试多种编码解码（UTF-8 优先，失败则 GBK/GB18030）
            text = None
            for enc in ("utf-8", "gbk", "gb18030"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                text = raw.decode("utf-8", errors="replace")
            data = json.loads(text)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                Translator._last_429_time = time.time()
            print(f"[LLM翻译] 失败，回退本地模型: {e}", flush=True)
            return None

    def batch_translate(self, texts, src_lang):
        """批量翻译：LLM 模式合并为一次请求避免限流，本地模式逐行翻译。"""
        if not texts:
            return texts

        # 确定目标语言
        if src_lang == config.TARGET_LANG:
            target_lang = config.FALLBACK_LANG
        else:
            target_lang = config.TARGET_LANG

        # LLM 模式：批量翻译，一次请求（冷却期内跳过）
        if config.TRANSLATION_ENGINE == "llm" and config.LLM_API_KEY and not self._in_cooldown():
            result = self._batch_translate_llm(texts, src_lang, target_lang)
            if result is not None:
                return result

        # 本地模式或 LLM 失败/冷却：直接用本地模型逐行翻译（不走 LLM）
        return [self._translate_local(t, src_lang) for t in texts]

    def _translate_local(self, text, src_lang):
        """仅用本地模型翻译（不尝试 LLM），用于批量回退。"""
        if src_lang == config.TARGET_LANG:
            target_lang = config.FALLBACK_LANG
        else:
            target_lang = config.TARGET_LANG

        if (src_lang, target_lang) in SPECIALIZED_MODELS:
            return self._translate_specialized(text, src_lang, target_lang)
        return self._translate_nllb(text, src_lang, target_lang)

    def _batch_translate_llm(self, texts, src_lang, tgt_lang):
        """LLM 批量翻译，用编号对应每一行。"""
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

        numbered_input = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        system_prompt = (
            f"你是一位专业的翻译专家。请将以下{src_name}文本逐行翻译成{tgt_name}。"
            f"要求：1.每行对应翻译，保持编号不变（如 '1. 译文'）；"
            f"2.只输出译文，不要解释；3.专业术语准确。"
        )
        user_prompt = f"请逐行翻译：\n{numbered_input}"

        try:
            resp = requests.post(
                config.LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {config.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=self._build_llm_body([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]),
                timeout=config.LLM_TIMEOUT * len(texts),
            )
            resp.raise_for_status()
            raw = resp.content
            for enc in ("utf-8", "gbk", "gb18030"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode("utf-8", errors="replace")
            data = json.loads(text)
            content = data["choices"][0]["message"]["content"].strip()

            # 解析编号行
            import re
            results = []
            lines = content.split("\n")
            for i in range(len(texts)):
                expected_prefix = f"{i+1}."
                found = False
                for line in lines:
                    line = line.strip()
                    if line.startswith(expected_prefix):
                        trans = line[len(expected_prefix):].strip()
                        # 去掉可能的引号包裹
                        if (trans.startswith('"') and trans.endswith('"')) or \
                           (trans.startswith("'") and trans.endswith("'")):
                            trans = trans[1:-1]
                        results.append(trans)
                        found = True
                        break
                if not found:
                    results.append(texts[i])  # 解析失败，保留原文
            return results
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                Translator._last_429_time = time.time()
            print(f"[LLM批量翻译] 失败，回退逐行: {e}", flush=True)
            return None

    def translate(self, text, src_lang):
        # 确定目标语言：源 != 目标 -> 目标语言；源 == 目标 -> 回退语言
        if src_lang == config.TARGET_LANG:
            target_lang = config.FALLBACK_LANG
        else:
            target_lang = config.TARGET_LANG

        # LLM 引擎优先（质量最高），冷却期内跳过，失败回退本地模型
        if config.TRANSLATION_ENGINE == "llm" and config.LLM_API_KEY and not self._in_cooldown():
            result = self._translate_llm(text, src_lang, target_lang)
            if result:
                return result

        # 本地引擎：日译中等有专用模型
        if (src_lang, target_lang) in SPECIALIZED_MODELS:
            return self._translate_specialized(text, src_lang, target_lang)

        # 其他走 NLLB
        return self._translate_nllb(text, src_lang, target_lang)
