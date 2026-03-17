"""Speech models: Whisper ASR (local + OpenAI API) and related constants.

All speech model classes implement: transcribe_audio_file(path: str) -> Optional[str]
so they can be used interchangeably (e.g. from audio_intent).
"""

import os
from typing import Any, Optional

WHISPER_MODEL_SIZE = "large-v3-turbo"
NO_SPEECH_PROB_MAX = 0.5
OPENAI_WHISPER_MODEL = "whisper-1"
DEEPGRAM_NOVA_MODEL = "nova-3"

# Optional noise reduction before transcription (noisereduce)
_noisereduce_available = False
try:
    import noisereduce
    _noisereduce_available = True
except ImportError:
    pass


def _reduce_noise_on_file(path: str) -> bool:
    """
    Reduce background noise in a WAV file in place using spectral gating.
    Returns True if denoising was applied, False if skipped (library missing or error).
    """
    if not _noisereduce_available:
        return False
    # Only process WAV files — skip WebM, MP3, etc. to avoid scipy.io.wavfile errors
    if not path.lower().endswith(".wav"):
        return False
    try:
        import numpy as np
        from scipy.io import wavfile
        sr, y = wavfile.read(path)
        if y.size == 0:
            return False
        if y.dtype != np.float32 and y.dtype != np.float64:
            y = y.astype(np.float32) / max(np.iinfo(y.dtype).max, 1)
        if y.ndim == 2:
            y = y.mean(axis=1)
        reduced = noisereduce.reduce_noise(
            y=y,
            sr=sr,
            stationary=True,
            prop_decrease=1.0,
        )
        out = (np.clip(reduced, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(path, sr, out)
        return True
    except Exception as e:
        print(f"[STT] noise reduction skipped: {e}")
        return False


class WhisperModel:
    """Whisper ASR model wrapper with lazy loading and configurable transcription."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = "cpu",
        no_speech_prob_max: float = NO_SPEECH_PROB_MAX,
    ):
        self._model_size = model_size
        self._device = device
        self._no_speech_prob_max = no_speech_prob_max
        self._model: Any = None

    @property
    def model(self):
        """Lazily load and return the underlying Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(
                self._model_size,
                device=self._device,
            )
        return self._model

    def transcribe_audio_file(self, path: str) -> Optional[str]:
        """
        Transcribe an audio file with Whisper.
        Returns cleaned text, or None if no confident speech segments.
        """
        _reduce_noise_on_file(path)
        result = self.model.transcribe(
            path,
            language="en",
            fp16=False,
            no_speech_threshold=self._no_speech_prob_max,
        )
        segments = result.get("segments", [])
        if segments:
            parts = [
                s["text"].strip()
                for s in segments
                if s.get("text", "").strip()
                and s.get("no_speech_prob", 1.0) <= self._no_speech_prob_max
            ]
            text = " ".join(parts).strip()
            if not text and segments:
                no_speech_probs = [s.get("no_speech_prob", 1.0) for s in segments[:5]]
                print(f"[Whisper] all {len(segments)} segment(s) filtered by no_speech_prob (threshold={self._no_speech_prob_max}); sample probs: {no_speech_probs}")
        else:
            text = (result.get("text") or "").strip()
            if not text:
                print(f"[Whisper] no segments in result for {path!r}; result.text={result.get('text')!r}")
        return text if text else None


class OpenAISTT:
    """OpenAI API speech-to-text (Whisper) for transcribing audio files."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_WHISPER_MODEL,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._client = None

    @property
    def client(self):
        """Lazily create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAISTT. Install with: pip install openai"
                ) from e
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY or pass api_key=..."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def transcribe_audio_file(self, path: str) -> Optional[str]:
        """
        Transcribe an audio file using OpenAI Whisper API.
        Returns cleaned text, or None if empty or on error.
        """
        _reduce_noise_on_file(path)
        try:
            with open(path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model=self._model,
                    file=f,
                    language="en",
                )
            text = (response.text or "").strip()
            return text if text else None
        except Exception:
            return None


class DeepgramNovaSTT:
    """Deepgram Nova-3 API speech-to-text for transcribing audio files."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEEPGRAM_NOVA_MODEL,
        smart_format: bool = True,
    ):
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self._model = model
        self._smart_format = smart_format
        self._client = None

    @property
    def client(self):
        """Lazily create Deepgram client."""
        if self._client is None:
            try:
                from deepgram import DeepgramClient
            except ImportError as e:
                raise ImportError(
                    "deepgram-sdk is required for DeepgramNovaSTT. Install with: pip install deepgram-sdk"
                ) from e
            if not self._api_key:
                raise ValueError(
                    "Deepgram API key required. Set DEEPGRAM_API_KEY or pass api_key=..."
                )
            self._client = DeepgramClient(api_key=self._api_key)
        return self._client

    def transcribe_audio_file(self, path: str) -> Optional[str]:
        """
        Transcribe an audio file using Deepgram Nova-3 API (listen.v1.media).
        Converts non-WAV files to WAV via ffmpeg if available, for reliable transcription.
        Returns cleaned text, or None if empty or on error.
        """
        import subprocess, tempfile
        # Convert non-WAV to WAV using ffmpeg for reliable Deepgram ingestion
        use_path = path
        tmp_wav = None
        if not path.lower().endswith(".wav"):
            try:
                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_wav.close()
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav.name],
                    capture_output=True, timeout=30,
                )
                if result.returncode == 0:
                    use_path = tmp_wav.name
                    print(f"[STT] converted {path} -> WAV ({os.path.getsize(tmp_wav.name)} bytes)")
                else:
                    print(f"[STT] ffmpeg conversion failed (rc={result.returncode}): {result.stderr[:200]}")
                    tmp_wav = None  # keep original path
            except FileNotFoundError:
                print("[STT] ffmpeg not found; sending raw audio to Deepgram")
                tmp_wav = None
            except Exception as e:
                print(f"[STT] ffmpeg conversion error: {e}")
                tmp_wav = None

        _reduce_noise_on_file(use_path)
        try:
            with open(use_path, "rb") as f:
                audio_bytes = f.read()
            print(f"[STT] Deepgram: sending {len(audio_bytes)} bytes from {use_path}")
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=self._model,
                language="en",
                smart_format=self._smart_format,
                punctuate=True,
            )
            # response is ListenV1Response: results.channels[0].alternatives[0].transcript
            results = getattr(response, "results", None)
            if not results:
                print(f"[STT] Deepgram: no results in response: {response}")
                return None
            channels = getattr(results, "channels", None) or []
            if not channels:
                print(f"[STT] Deepgram: no channels in results: {results}")
                return None
            alternatives = getattr(channels[0], "alternatives", None) or []
            if not alternatives:
                print(f"[STT] Deepgram: no alternatives in channel: {channels[0]}")
                return None
            text = (getattr(alternatives[0], "transcript", None) or "").strip()
            confidence = getattr(alternatives[0], "confidence", None)
            print(f"[STT] Deepgram: transcript={text!r}, confidence={confidence}")
            return text if text else None
        except Exception as e:
            print(f"[STT] Deepgram transcribe_file error: {e}")
            return None
        finally:
            if tmp_wav and os.path.exists(tmp_wav.name):
                try:
                    os.unlink(tmp_wav.name)
                except Exception:
                    pass
