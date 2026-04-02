from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────────
# Platform Detection
# ─────────────────────────────────────────────────────────────────────
IS_RASPBERRY_PI = (
    os.path.exists("/etc/rpi-issue") or
    os.path.exists("/proc/device-tree/model")
)


# ─────────────────────────────────────────────────────────────────────
# General
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GeneralSettings:
    log_level:                str = os.getenv("ROBOT_LOG_LEVEL", "INFO")
    student_name:             str = os.getenv("ROBOT_STUDENT_NAME", "Student")
    default_session_language: str = os.getenv("ROBOT_DEFAULT_SESSION_LANGUAGE", "ar")


# ─────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CameraSettings:
    width:        int = int(os.getenv("ROBOT_CAM_WIDTH",  "1280"))
    height:       int = int(os.getenv("ROBOT_CAM_HEIGHT", "720"))
    fps:          int = int(os.getenv("ROBOT_CAM_FPS",    "30"))
    index:        int = int(os.getenv("ROBOT_CAM_INDEX",  "0"))   # OpenCV device index
    format:       str = os.getenv("ROBOT_CAM_FORMAT", "BGR")      # numpy array format for all modules
    buffer_size:  int = int(os.getenv("ROBOT_CAM_BUFFER", "1"))   # keep only latest frame


# ─────────────────────────────────────────────────────────────────────
# ASR  (from voice pipeline)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ASRSettings:
    provider:                        str   = os.getenv("ROBOT_ASR_PROVIDER", "google")
    sample_rate:                     int   = int(os.getenv("ROBOT_ASR_SAMPLE_RATE", "16000"))
    language_mode:                   str   = os.getenv("ROBOT_ASR_LANGUAGE_MODE", "auto")
    default_record_duration_seconds: float = float(os.getenv("ROBOT_ASR_DEFAULT_DURATION_SEC", "5.0"))
    supported_languages: Dict[str, str]    = field(default_factory=lambda: {"en": "en-US", "ar": "ar-EG"})


# ─────────────────────────────────────────────────────────────────────
# VAD  (from voice pipeline)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VADSettings:
    sample_rate:               int   = int(os.getenv("ROBOT_VAD_SAMPLE_RATE",        "16000"))
    initial_threshold:         float = float(os.getenv("ROBOT_VAD_THRESHOLD",        "0.60"))
    chunk_duration_ms:         int   = int(os.getenv("ROBOT_VAD_CHUNK_MS",           "32"))
    pre_speech_buffer_seconds: float = float(os.getenv("ROBOT_VAD_PRE_ROLL_SEC",     "0.30"))
    min_speech_seconds:        float = float(os.getenv("ROBOT_VAD_MIN_SPEECH_SEC",   "0.50"))
    silence_timeout_seconds:   float = float(os.getenv("ROBOT_VAD_SILENCE_TIMEOUT_SEC", "0.80"))
    max_abs_amplitude:         float = float(os.getenv("ROBOT_VAD_MAX_ABS_AMP",      "1.0"))
    torch_threads:             int   = int(os.getenv("ROBOT_VAD_TORCH_THREADS",      "1"))
    model_local_path:          str   = os.getenv("ROBOT_VAD_MODEL_LOCAL_PATH",       "")
    model_hub_repo:            str   = os.getenv("ROBOT_VAD_HUB_REPO",     "snakers4/silero-vad")
    model_hub_name:            str   = os.getenv("ROBOT_VAD_HUB_NAME",     "silero_vad")
    model_trust_repo:          bool  = os.getenv("ROBOT_VAD_TRUST_REPO", "true").lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────
# LLM  (OpenRouter)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMSettings:
    provider:                              str = os.getenv("ROBOT_LLM_PROVIDER", "openrouter")
    openrouter_api_key:                    str = os.getenv("ROBOT_OPENROUTER_API_KEY", "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")
    openrouter_model:                      str = os.getenv("ROBOT_OPENROUTER_MODEL", "qwen/qwen3-30b-a3b:free")
    openrouter_availability_timeout_seconds: int = int(os.getenv("ROBOT_OPENROUTER_AVAILABILITY_TIMEOUT_SEC", "5"))
    request_timeout_seconds:               int = int(os.getenv("ROBOT_LLM_REQUEST_TIMEOUT_SEC",    "90"))
    summarization_timeout_seconds:         int = int(os.getenv("ROBOT_LLM_SUMMARIZE_TIMEOUT_SEC",  "60"))
    db_path:                               str = os.getenv("ROBOT_LLM_DB_PATH", "robot_sessions.db")
    sliding_window_size:                   int = int(os.getenv("ROBOT_LLM_WINDOW_SIZE", "10"))

    system_prompt_arabic: str = (
        "أنت روبوت تعليمي ذكي مساعد للطلاب في رحلتهم التعليمية.\n\n"
        "**أهدافك الرئيسية:**\n"
        "1. شرح المفاهيم بطريقة بسيطة وممتعة\n"
        "2. تشجيع الفضول والأسئلة\n"
        "3. مساعدة الطالب على فهم الدروس\n"
        "4. توفير أمثلة عملية ذات صلة\n\n"
        "**قواعد التفاعل:**\n"
        "- تحدث بلغة عربية فصحى مع لمسات من اللهجة المصرية\n"
        "- اجعل الإجابات قصيرة وسهلة الفهم (جملتين لثلاث جمل كحد أقصى)\n"
        "- إذا لم تفهم السؤال، اطلب توضيح برفق\n"
        "- استخدم تشبيهات وأمثلة من الحياة اليومية\n"
        "- كن متحمساً وإيجابياً دائماً\n\n"
        "**لما بتلاقي وصف بصري في رسالة [VISION]:**\n"
        "- استخدمه كمحتوى تعليمي طبيعي\n"
        "- لو في emotion → رد بشكل طبيعي ومناسب للحالة\n"
        "- لو في objects → اربطهم بالدرس أو اسأل سؤال تعليمي عنهم\n"
        "- متقولش 'لقيت في الوصف' أو 'حسب الصورة' — تصرف بشكل طبيعي\n\n"
        "**ممنوع:**\n"
        "- إعطاء الإجابة الكاملة مباشرة\n"
        "- الإجابات الطويلة جداً\n"
        "- استخدام مصطلحات معقدة بدون شرح"
    )

    system_prompt_english: str = (
        "You are an intelligent educational robot assistant helping students in their learning journey.\n\n"
        "**Your main goals:**\n"
        "1. Explain concepts in a simple and engaging way\n"
        "2. Encourage curiosity and questions\n"
        "3. Help students understand lessons\n"
        "4. Provide practical relevant examples\n\n"
        "**Interaction rules:**\n"
        "- Speak in clear, simple English\n"
        "- Keep answers short and easy to understand (2-3 sentences max)\n"
        "- Use analogies and examples from daily life\n"
        "- Be enthusiastic and positive always\n\n"
        "**When you find a visual description in a [VISION] message:**\n"
        "- Use it as natural educational content\n"
        "- If emotion is present → respond naturally and appropriately\n"
        "- If objects are present → connect them to the lesson or ask an educational question\n"
        "- Never say 'according to the description' — act naturally\n\n"
        "**Forbidden:**\n"
        "- Giving complete answers directly\n"
        "- Very long responses\n"
        "- Using complex terms without explanation"
    )


# ─────────────────────────────────────────────────────────────────────
# TTS  (edge_tts + pygame)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TTSSettings:
    engine:                       str   = os.getenv("ROBOT_TTS_ENGINE", "edge_tts")
    audio_temp_dir:               str   = os.getenv("ROBOT_TTS_TEMP_DIR", tempfile.gettempdir())
    audio_filename_template:      str   = os.getenv("ROBOT_TTS_AUDIO_TEMPLATE", "tts_{turn_id}.mp3")
    pygame_poll_interval_seconds: float = float(os.getenv("ROBOT_TTS_POLL_SEC", "0.05"))
    voice_map: Dict[str, str]           = field(default_factory=lambda: {
        "en": "en-US-GuyNeural",
        "ar": "ar-EG-SalmaNeural",
    })


# ─────────────────────────────────────────────────────────────────────
# Root Settings
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Settings:
    general: GeneralSettings
    camera:  CameraSettings
    asr:     ASRSettings
    vad:     VADSettings
    llm:     LLMSettings
    tts:     TTSSettings


def get_settings() -> Settings:
    return Settings(
        general = GeneralSettings(),
        camera  = CameraSettings(),
        asr     = ASRSettings(),
        vad     = VADSettings(),
        llm     = LLMSettings(),
        tts     = TTSSettings(),
    )