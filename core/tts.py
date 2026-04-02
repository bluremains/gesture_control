"""
core/tts.py — TTS Module (edge_tts + pygame)

Thread-safe, interruptible text-to-speech pipeline.
- Auto language detection (Arabic / English)
- stop() halts audio immediately
- Concurrent speak() calls are serialized via request_id
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from config.settings import get_settings

_TTS = get_settings().tts
logger = logging.getLogger(__name__)


class TTSModuleError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Return 'ar' if text contains Arabic characters, else 'en'."""
    for char in text:
        if "\u0600" <= char <= "\u06FF":
            return "ar"
    return "en"


# ─────────────────────────────────────────────────────────────────────
# TTSModule
# ─────────────────────────────────────────────────────────────────────
class TTSModule:
    """
    Usage:
        tts = TTSModule()
        tts.speak("مرحباً يا طالب!")       # auto-detects Arabic
        tts.speak("Hello!", language="en")
        tts.stop()                          # interrupt immediately
    """

    def __init__(self):
        self._stop_event  = threading.Event()
        self._lock        = threading.Lock()
        self._request_id  = 0
        self._play_thread: Optional[threading.Thread] = None

        self._pygame       = None
        self._edge_tts     = None
        self._pygame_ready = False

        if getattr(_TTS, "engine", "edge_tts") != "edge_tts":
            raise TTSModuleError(
                f"[tts] Unsupported engine '{_TTS.engine}'. Only 'edge_tts' is supported."
            )

        self._init_dependencies()

    # ── Init ─────────────────────────────────────────────────────────

    def _init_dependencies(self):
        try:
            import pygame
            self._pygame = pygame
        except Exception as e:
            raise TTSModuleError(
                "[tts] pygame not found. Install it: pip install pygame"
            ) from e

        try:
            import edge_tts
            self._edge_tts = edge_tts
        except Exception as e:
            raise TTSModuleError(
                "[tts] edge_tts not found. Install it: pip install edge-tts"
            ) from e

        try:
            self._pygame.mixer.init()
            self._pygame_ready = True
        except Exception as e:
            raise TTSModuleError(
                "[tts] pygame mixer init failed — check audio device."
            ) from e

    # ── Public API ───────────────────────────────────────────────────

    def is_playing(self) -> bool:
        if not self._pygame_ready:
            return False
        try:
            return bool(self._pygame.mixer.music.get_busy())
        except Exception:
            return False

    def stop(self):
        """Stop current playback immediately and cancel any pending requests."""
        with self._lock:
            self._stop_event.set()
            self._request_id += 1

        try:
            if self._pygame_ready:
                self._pygame.mixer.music.stop()
                self._pygame.mixer.music.unload()
        except Exception:
            pass

    def speak(self, text: str, language: Optional[str] = None):
        """
        Generate and play speech in a background thread.
        Returns immediately — use stop() to interrupt.

        Args:
            text:     text to speak
            language: 'ar' or 'en' — auto-detected if None
        """
        if not text or not text.strip():
            raise TTSModuleError("[tts.speak] text must not be empty")

        if language is None:
            language = detect_language(text)

        voice = _TTS.voice_map.get(language, _TTS.voice_map.get("en", "en-US-GuyNeural"))

        with self._lock:
            self._stop_event.clear()
            self._request_id += 1
            request_id = self._request_id

        path = self._temp_path(request_id)

        def worker():
            try:
                if self._stop_event.is_set():
                    return

                # Generate audio file
                asyncio.run(self._generate(text, voice, path))

                # Check if still valid before playing
                with self._lock:
                    if request_id != self._request_id or self._stop_event.is_set():
                        self._delete_file(path)
                        return

                self._play(path, request_id)

            except Exception as e:
                logger.error(f"[TTS] speak() worker error: {e}", exc_info=True)

        self._play_thread = threading.Thread(target=worker, daemon=True)
        self._play_thread.start()

    # ── Internal ─────────────────────────────────────────────────────

    def _temp_path(self, request_id: int) -> str:
        os.makedirs(_TTS.audio_temp_dir, exist_ok=True)
        filename = _TTS.audio_filename_template.format(turn_id=request_id)
        return os.path.join(_TTS.audio_temp_dir, filename)

    async def _generate(self, text: str, voice: str, path: str):
        communicate = self._edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(path)

    def _play(self, path: str, request_id: int):
        try:
            if self._stop_event.is_set():
                return

            self._pygame.mixer.music.load(path)
            self._pygame.mixer.music.play()

            while self._pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    return
                time.sleep(_TTS.pygame_poll_interval_seconds)

        except Exception as e:
            logger.error(f"[TTS] Playback error: {e}", exc_info=True)
        finally:
            try:
                self._pygame.mixer.music.unload()
            except Exception:
                pass
            self._delete_file(path)

    @staticmethod
    def _delete_file(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass