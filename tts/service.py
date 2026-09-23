"""Non-blocking queue-based Text-to-Speech service using pyttsx3 and COM initialization."""

import os
import queue
import sys
import threading
from typing import Optional
from core.logger import logger


class TTSService:
    """Manages asynchronous speech synthesis in a background thread."""

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 150,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
    ):
        self.enabled = enabled and (os.getenv("TTS_ENABLED", "true").lower() == "true")
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id or os.getenv("TTS_VOICE_ID", "")
        self.queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TTSWorkerThread")
        self._thread.start()
        logger.info("TTSService worker thread initialized.")

    def _worker_loop(self) -> None:
        # Initialize COM on Windows OS inside worker thread
        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception as e:
                logger.warning(f"TTSService COM initialization warning: {e}")

        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
        except Exception as e:
            logger.warning(f"TTSService pyttsx3 engine initialization failed: {e}. TTS disabled.")
            self.enabled = False
            return

        while self._running:
            try:
                text = self.queue.get(timeout=0.5)
                if text is None:
                    break
                logger.debug(f"TTSService speaking: '{text[:40]}...'")
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTSService speech error: {e}")

        if engine:
            try:
                engine.stop()
            except Exception:
                pass

        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        """Enqueues text for non-blocking background speech."""
        if not self.enabled or not text.strip():
            return
        self.queue.put(text)

    def speak_blocking(self, text: str) -> None:
        """Enqueues text and waits for audio speech completion."""
        if not self.enabled or not text.strip():
            return
        self.queue.put(text)
        self.queue.join()

    def shutdown(self) -> None:
        """Shuts down the background TTS worker."""
        if not self.enabled:
            return
        self._running = False
        self.queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("TTSService stopped.")
