"""
core/llm.py — LLM Module (OpenRouter + Memory + Vision Context)

Handles:
- OpenRouter API communication
- Sliding window conversation memory (SQLite)
- Auto-summarization every N messages
- vision_context injection from vision modules
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
_LLM      = _SETTINGS.llm


# ─────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────
class LLMModuleError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────
# Vision Context Builder
# ─────────────────────────────────────────────────────────────────────
def build_vision_prompt(user_message: str, vision_context: Optional[Dict[str, Any]]) -> str:
    """
    Merge user message with vision context into a single prompt string.

    vision_context can carry keys from any active vision module:
        objects   → list[str]        from object_recognition
        scene     → str              from scene_segmentation
        emotion   → str              from emotion_detection
        face      → str              "same_student" | "new_student"
        gesture   → str              from gesture_control

    Example output injected into the message:
        [VISION] شايف: كتاب، قلم | طالب زعلان
        السؤال: ايه الفرق بين الفيزياء والكيمياء؟
    """
    if not vision_context:
        return user_message

    parts = []

    if vision_context.get("objects"):
        objects_str = "، ".join(vision_context["objects"])
        parts.append(f"شايف: {objects_str}")

    if vision_context.get("scene"):
        parts.append(f"المشهد: {vision_context['scene']}")

    if vision_context.get("emotion"):
        emotion_map = {
            "happy":    "الطالب سعيد",
            "sad":      "الطالب حزين",
            "angry":    "الطالب زعلان",
            "neutral":  "الطالب هادي",
            "fear":     "الطالب خايف",
            "disgust":  "الطالب مش مرتاح",
            "surprise": "الطالب متفاجئ",
        }
        parts.append(emotion_map.get(vision_context["emotion"], vision_context["emotion"]))

    if vision_context.get("face") == "new_student":
        parts.append("طالب جديد")

    if not parts:
        return user_message

    vision_line = "[VISION] " + " | ".join(parts)
    return f"{vision_line}\nالسؤال: {user_message}" if user_message else vision_line


# ─────────────────────────────────────────────────────────────────────
# OpenRouter Connection
# ─────────────────────────────────────────────────────────────────────
class OpenRouterConnection:
    """Low-level OpenRouter API wrapper."""

    def __init__(
        self,
        api_key:                      Optional[str] = None,
        model:                        Optional[str] = None,
        chat_timeout_seconds:         Optional[int] = None,
        availability_timeout_seconds: Optional[int] = None,
    ):
        self.api_key      = api_key  or _LLM.openrouter_api_key
        self.model        = model    or _LLM.openrouter_model
        self.chat_timeout = chat_timeout_seconds         or _LLM.request_timeout_seconds
        self.avail_timeout = availability_timeout_seconds or _LLM.openrouter_availability_timeout_seconds
        self.base_url     = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise LLMModuleError(
                "[llm] Missing OpenRouter API key. "
                "Set ROBOT_OPENROUTER_API_KEY env variable."
            )

    def is_available(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.avail_timeout,
            )
            return r.status_code == 200
        except Exception as e:
            logger.error(f"[LLM] OpenRouter not reachable: {e}")
            return False

    def chat(self, messages: List[dict], timeout: Optional[int] = None) -> str:
        timeout = timeout or self.chat_timeout
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages},
                headers={
                    "Authorization":  f"Bearer {self.api_key}",
                    "Content-Type":   "application/json",
                },
                timeout=timeout,
            )

            if response.status_code != 200:
                raise LLMModuleError(
                    f"[llm] OpenRouter returned {response.status_code}: {response.text}"
                )

            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                raise LLMModuleError("[llm] Empty response from OpenRouter")

            return content

        except requests.Timeout:
            raise LLMModuleError(f"[llm] OpenRouter timeout after {timeout}s")
        except requests.ConnectionError:
            raise LLMModuleError("[llm] Cannot reach OpenRouter — check internet connection")
        except LLMModuleError:
            raise
        except Exception as e:
            raise LLMModuleError(f"[llm] Unexpected error: {e}") from e


# ─────────────────────────────────────────────────────────────────────
# Session Manager
# ─────────────────────────────────────────────────────────────────────
class SessionManager:
    """SQLite-backed session and message storage."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _LLM.db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id   TEXT PRIMARY KEY,
                        student_name TEXT NOT NULL,
                        language     TEXT NOT NULL,
                        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id  TEXT NOT NULL,
                        role        TEXT NOT NULL,
                        content     TEXT NOT NULL,
                        timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    );
                    CREATE TABLE IF NOT EXISTS summaries (
                        summary_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id    TEXT NOT NULL,
                        summary_text  TEXT NOT NULL,
                        message_count INTEGER NOT NULL,
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    );
                """)
            logger.info("[LLM] Database ready")
        except Exception as e:
            raise LLMModuleError(f"[llm] DB init failed: {e}") from e

    def create_session(self, student_name: str, language: Optional[str] = None) -> str:
        if not student_name:
            raise LLMModuleError("[llm] student_name must be a non-empty string")

        language = language or _SETTINGS.general.default_session_language
        if language not in ("ar", "en"):
            raise LLMModuleError("[llm] language must be 'ar' or 'en'")

        session_id = f"{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, student_name, language) VALUES (?,?,?)",
                    (session_id, student_name, language)
                )
            logger.info(f"[LLM] Session created: {session_id}")
            return session_id
        except Exception as e:
            raise LLMModuleError(f"[llm] create_session DB error: {e}") from e

    def add_message(self, session_id: str, role: str, content: str) -> int:
        if role not in ("user", "assistant"):
            raise LLMModuleError("[llm] role must be 'user' or 'assistant'")
        if not content:
            raise LLMModuleError("[llm] content must be non-empty")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                    (session_id, role, content)
                )
            return cursor.lastrowid
        except Exception as e:
            raise LLMModuleError(f"[llm] add_message DB error: {e}") from e

    def get_sliding_window(self, session_id: str, window_size: int = 10) -> List[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT role, content FROM messages "
                    "WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, window_size)
                ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        except Exception as e:
            raise LLMModuleError(f"[llm] get_sliding_window DB error: {e}") from e

    def get_full_history(self, session_id: str) -> List[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT role, content FROM messages "
                    "WHERE session_id=? ORDER BY timestamp ASC",
                    (session_id,)
                ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as e:
            raise LLMModuleError(f"[llm] get_full_history DB error: {e}") from e

    def get_message_count(self, session_id: str) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
                ).fetchone()[0]
        except Exception as e:
            raise LLMModuleError(f"[llm] get_message_count DB error: {e}") from e

    def save_summary(self, session_id: str, summary_text: str, message_count: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO summaries (session_id, summary_text, message_count) VALUES (?,?,?)",
                    (session_id, summary_text, message_count)
                )
            logger.info(f"[LLM] Summary saved ({message_count} messages)")
        except Exception as e:
            raise LLMModuleError(f"[llm] save_summary DB error: {e}") from e

    def get_session_language(self, session_id: str) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT language FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            return result[0] if result else _SETTINGS.general.default_session_language
        except Exception:
            return _SETTINGS.general.default_session_language


# ─────────────────────────────────────────────────────────────────────
# Memory Manager
# ─────────────────────────────────────────────────────────────────────
class MemoryManager:
    """Sliding window + auto-summarization."""

    def __init__(
        self,
        session_manager: SessionManager,
        openrouter:      OpenRouterConnection,
        window_size:     Optional[int] = None,
    ):
        self.session_manager = session_manager
        self.openrouter      = openrouter
        self.window_size     = int(window_size or _LLM.sliding_window_size)

    def should_summarize(self, session_id: str) -> bool:
        return self.session_manager.get_message_count(session_id) >= self.window_size

    def summarize(self, session_id: str) -> Optional[str]:
        history  = self.session_manager.get_full_history(session_id)
        language = self.session_manager.get_session_language(session_id)
        if not history:
            return None

        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

        if language == "ar":
            prompt = f"يرجى تلخيص المحادثة التعليمية التالية بشكل موجز:\n\n{conv_text}\n\nالملخص:"
        else:
            prompt = f"Please summarize the following educational conversation concisely:\n\n{conv_text}\n\nSummary:"

        try:
            logger.info("[LLM] Summarizing conversation...")
            summary = self.openrouter.chat(
                [{"role": "user", "content": prompt}],
                timeout=_LLM.summarization_timeout_seconds,
            )
            if summary:
                self.session_manager.save_summary(session_id, summary, len(history))
            return summary
        except Exception as e:
            logger.error(f"[LLM] Summarization failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────
# LLM Module  (main entry point)
# ─────────────────────────────────────────────────────────────────────
class LLMModule:
    """
    Main interface for the LLM pipeline.

    Usage:
        llm = LLMModule()
        session_id = llm.create_session("Ahmed")

        # plain voice message
        response = llm.chat(session_id, "ايه هو الضوء؟")

        # with vision context from vision modules
        response = llm.chat(
            session_id,
            "ايه ده؟",
            vision_context={
                "objects": ["كتاب", "قلم"],
                "emotion": "confused",
            }
        )

        llm.tts_speak(response)   # optional: pipe to TTS directly
    """

    def __init__(
        self,
        backend:         Optional[OpenRouterConnection] = None,
        session_manager: Optional[SessionManager]       = None,
    ):
        self.openrouter      = backend         or OpenRouterConnection()
        self.session_manager = session_manager or SessionManager()
        self.memory_manager  = MemoryManager(self.session_manager, self.openrouter)

    # ── Session ──────────────────────────────────────────────────────

    def create_session(self, student_name: str, language: Optional[str] = None) -> str:
        """Create a new student session. Returns session_id."""
        return self.session_manager.create_session(student_name, language)

    def is_ready(self) -> bool:
        """Returns True if OpenRouter API is reachable."""
        return self.openrouter.is_available()

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        session_id:     str,
        user_message:   str,
        vision_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            session_id:     from create_session()
            user_message:   text from ASR or direct input
            vision_context: optional dict from any vision module, e.g.:
                            {"objects": ["كتاب"], "emotion": "happy"}

        Returns:
            Assistant response string.

        Raises:
            LLMModuleError on failure.
        """
        if not user_message and not vision_context:
            raise LLMModuleError("[llm.chat] user_message and vision_context cannot both be empty")

        language      = self.session_manager.get_session_language(session_id)
        system_prompt = (
            _LLM.system_prompt_arabic if language == "ar"
            else _LLM.system_prompt_english
        )

        # Build final message — merge user text + vision context
        final_message = build_vision_prompt(user_message, vision_context)

        # Save raw user message to DB (not the vision-enriched version)
        self.session_manager.add_message(session_id, "user", user_message or "[vision update]")

        # Sliding window history
        history = self.session_manager.get_sliding_window(
            session_id, window_size=self.memory_manager.window_size
        )

        messages = [
            {"role": "system",  "content": system_prompt},
            *history,
            {"role": "user",    "content": final_message},
        ]

        logger.info(f"[LLM] Sending to OpenRouter (lang={language}, vision={bool(vision_context)})")
        response = self.openrouter.chat(messages)

        self.session_manager.add_message(session_id, "assistant", response)
        logger.info("[LLM] Response received and saved")

        # Auto-summarize if threshold reached
        if self.memory_manager.should_summarize(session_id):
            self.memory_manager.summarize(session_id)

        return response