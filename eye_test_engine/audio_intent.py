"""
Convert patient audio to intent by speech-to-text and mapping transcript to session options.
Uses a pluggable speech model from speech_models (WhisperModel, OpenAISTT, DeepgramNovaSTT).
Uses sentence_transformers for semantic text→intent matching when available.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# Sentence transformer for semantic matching (lazy-loaded)
_sentence_model = None
_SENTENCE_MODEL_NAME = os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2")
# Minimum cosine similarity to accept a match (0–1)
_SIMILARITY_THRESHOLD = float(os.environ.get("AUDIO_INTENT_SIM_THRESHOLD", "0.35"))

# Pluggable speech model from speech_models.py (lazy-loaded)
# Env: SPEECH_MODEL = whisper | openai | deepgram
_SPEECH_MODEL_NAME = os.environ.get("SPEECH_MODEL", "whisper").strip().lower()
_speech_model_instance: Optional[object] = None


def _get_speech_model() -> Optional[object]:
    """Return the configured speech model instance (from speech_models). Cached per process."""
    global _speech_model_instance
    if _speech_model_instance is not None:
        return _speech_model_instance
    if not _HAS_SPEECH_MODELS:
        return None
    try:
        from speech_models import WhisperModel as SMWhisper
        from speech_models import OpenAISTT
        from speech_models import DeepgramNovaSTT

        if _SPEECH_MODEL_NAME == "openai":
            _speech_model_instance = OpenAISTT()
        elif _SPEECH_MODEL_NAME == "deepgram":
            _speech_model_instance = DeepgramNovaSTT()
        else:
            # "whisper" or any other value: use local Whisper
            no_speech_max = float(os.environ.get("WHISPER_NO_SPEECH_PROB_MAX", "0.5"))
            _speech_model_instance = SMWhisper(
                model_size=os.environ.get("WHISPER_MODEL", "base"),
                device=os.environ.get("WHISPER_DEVICE", "cpu"),
                no_speech_prob_max=no_speech_max,
            )
        return _speech_model_instance
    except Exception:
        return None


try:
    from speech_models import WhisperModel  # noqa: F401
    from speech_models import OpenAISTT  # noqa: F401
    from speech_models import DeepgramNovaSTT  # noqa: F401
    _HAS_SPEECH_MODELS = True
except ImportError:
    _HAS_SPEECH_MODELS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except ImportError:
    _HAS_PYDUB = False


def _get_sentence_model():
    """Lazy-load the sentence transformer model for semantic matching."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        _sentence_model = SentenceTransformer(_SENTENCE_MODEL_NAME)
        return _sentence_model
    except Exception:
        return None


def _normalize(s: str) -> str:
    """Lowercase, collapse spaces, remove punctuation for matching."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Common phrase → canonical intent keywords (for fuzzy match)
_PHRASE_TO_KEYWORDS = {
    "readable": ["readable", "read", "clear", "yes", "can see", "see it", "good"],
    "not_readable": ["not readable", "can't read", "cannot read", "unable to read", "no", "can't see"],
    "blurry": ["blurry", "blur", "blurred", "fuzzy"],
    "better_1": ["one", "first", "1", "first one", "first is better", "one is better", "option one", "flip one"],
    "better_2": ["two", "second", "2", "second one", "second is better", "two is better", "option two", "flip two"],
    "same": ["same", "equal", "both", "same thing", "no difference", "either"],
    "cant_tell": ["can't tell", "cannot tell", "cant tell", "don't know", "dont know", "not sure", "unsure"],
    "red_clearer": ["red", "red is clearer", "red one"],
    "green_clearer": ["green", "green is clearer", "green one"],
    "top_clearer": ["top", "top is clearer", "top one"],
    "bottom_clearer": ["bottom", "bottom is clearer", "bottom one"],
    "target_ok": ["ok", "okay", "good", "clear", "comfortable", "target ok"],
    "not_clear": ["not clear", "not comfortable", "unclear"],
}


def _option_to_key(opt: str) -> Optional[str]:
    """Map option label to key for _PHRASE_TO_KEYWORDS."""
    n = _normalize(opt.replace("_", " "))
    for key, keywords in _PHRASE_TO_KEYWORDS.items():
        if key.replace("_", " ") in n or any(kw in n for kw in keywords):
            return key
    # By prefix
    if n.startswith("readable") and "not" not in n:
        return "readable"
    if "not readable" in n or "unable" in n:
        return "not_readable"
    if "blur" in n:
        return "blurry"
    if "better" in n and ("1" in n or "one" in n or "first" in n):
        return "better_1"
    if "better" in n and ("2" in n or "two" in n or "second" in n):
        return "better_2"
    if "same" in n or "equal" in n:
        return "same"
    if "can't tell" in n or "cant tell" in n or "don't know" in n:
        return "cant_tell"
    if "red" in n and "clear" in n:
        return "red_clearer"
    if "green" in n and "clear" in n:
        return "green_clearer"
    if "top" in n and "clear" in n:
        return "top_clearer"
    if "bottom" in n and "clear" in n:
        return "bottom_clearer"
    if "target" in n and "ok" in n:
        return "target_ok"
    if "not clear" in n:
        return "not_clear"
    return None


def _score_match(transcript: str, option: str, index: int) -> float:
    """
    Score how well transcript matches this option (0 = no match, 1 = exact).
    Uses keyword overlap and position (option 1, 2, 3...).
    """
    t = _normalize(transcript)
    o = _normalize(option.replace("_", " "))
    if not t:
        return 0.0

    # Exact or substring match
    if o in t or t in o:
        return 0.95
    if o and t and o.split()[0] in t:
        return 0.7

    # Number word match: "one"/"first"/"1" -> option 1
    number_words = [
        ["one", "first", "1", "option one", "first one"],
        ["two", "second", "2", "option two", "second one"],
        ["three", "third", "3", "option three"],
        ["four", "fourth", "4", "option four"],
        ["five", "fifth", "5", "option five"],
        ["six", "sixth", "6", "option six"],
    ]
    for i, words in enumerate(number_words):
        if i == index and any(w in t for w in words):
            return 0.85

    # Keyword match via canonical key
    key = _option_to_key(option)
    if key and key in _PHRASE_TO_KEYWORDS:
        for kw in _PHRASE_TO_KEYWORDS[key]:
            if kw in t:
                return 0.9

    # Word overlap
    t_words = set(t.split())
    o_words = set(o.split())
    overlap = len(t_words & o_words) / max(len(o_words), 1)
    if overlap >= 0.5:
        return 0.6 + overlap * 0.3
    return 0.0


def _score_with_sentence_transformers(transcript: str, options: List[str]) -> Optional[Tuple[str, float]]:
    """
    Use sentence_transformers to compute semantic similarity between transcript and each option.
    Returns (best_option, similarity) or None if model unavailable or no score above threshold.
    """
    model = _get_sentence_model()
    if model is None or not transcript.strip() or not options:
        return None
    try:
        from sentence_transformers import util
        transcript_emb = model.encode(transcript.strip(), convert_to_tensor=True)
        option_embeddings = model.encode([o.replace("_", " ") for o in options], convert_to_tensor=True)
        scores = util.cos_sim(transcript_emb.reshape(1, -1), option_embeddings)[0]
        best_idx = int(scores.argmax().item())
        best_score = float(scores[best_idx].item())
        if best_score >= _SIMILARITY_THRESHOLD:
            return (options[best_idx], min(1.0, max(0.0, best_score)))
    except Exception:
        pass
    return None


def _is_chart_reading_phase(phase_name: Optional[str]) -> bool:
    """True when phase is spherical refinement or distance vision (Snellen chart reading)."""
    if not phase_name:
        return False
    p = (phase_name or "").lower()
    return "sphere" in p or "coarse sphere" in p or "distance baseline" in p or "distance vision" in p


def _chart_intent_to_option(chart_intent: str, options: List[str]) -> Optional[str]:
    """Map chart reading intent (READABLE/BLURRY/NOT_READABLE) to an option from the session."""
    if not chart_intent or not options:
        return None
    want = chart_intent.upper().replace(" ", "_")
    for opt in options:
        if (opt or "").upper().replace(" ", "_") == want:
            return opt
    return None


def transcript_to_intent(
    transcript: str,
    options: List[str],
    *,
    phase_name: Optional[str] = None,
    chart_param: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """
    Map transcript to the best matching option from the current session.
    For spherical refinement or distance vision phases, uses chart_reading.get_chart_intend.
    Otherwise uses sentence_transformers for semantic matching when available; else keyword-based scoring.
    Returns (matched_option_string, confidence) or None if no good match.
    """
    if not transcript or not options:
        return None
    # Fix common STT mishearings
    transcript = re.sub(r"\bbloody\b", "blurry", transcript, flags=re.IGNORECASE)
    transcript = re.sub(r"\bdo\b", "two", transcript, flags=re.IGNORECASE)
    t = _normalize(transcript)
    if not t:
        return None

    # Spherical refinement or distance vision: use chart reading intent
    if _is_chart_reading_phase(phase_name) and chart_param:
        try:
            from chart_reading import CHART_LETTERS_MAP, ChartReadingDetector

            letters_val = CHART_LETTERS_MAP.get(str(chart_param).strip())
            if letters_val is not None:
                chart_letters = "".join(letters_val) if isinstance(letters_val, list) else letters_val
                detector = ChartReadingDetector()
                chart_intent = detector.get_chart_intend(options, transcript, chart_letters)
                matched = _chart_intent_to_option(chart_intent, options)
                if matched is not None:
                    return (matched, 0.9)
        except Exception:
            pass  # fall through to normal matching

    # Map "Yes" / "No" to Readable / Not Readable when those options are present
    if t == "yes" and "READABLE" in options:
        return ("READABLE", 1.0)
    if t == "no" and "NOT_READABLE" in options:
        return ("NOT_READABLE", 1.0)

    result = _score_with_sentence_transformers(transcript, options)
    if result is not None:
        return result

    best_option = None
    best_score = 0.0
    for i, opt in enumerate(options):
        score = _score_match(transcript, opt, i)
        if score > best_score:
            best_score = score
            best_option = opt

    if best_option and best_score >= 0.5:
        return (best_option, best_score)
    return None


def speech_to_text_available() -> bool:
    """True if a speech model from speech_models is available and STT can run."""
    return _HAS_SPEECH_MODELS


def audio_to_transcript(audio_data: bytes, content_type: str = "") -> str:
    """
    Run speech-to-text on audio bytes using the configured speech model (speech_models).
    Supports WAV, webm, mp3 (webm/mp3 need pydub/ffmpeg).
    Returns empty string on failure or if dependencies missing.
    """
    if not _HAS_SPEECH_MODELS:
        raise RuntimeError(
            "Speech models not available. Install speech_models dependencies (e.g. whisper, openai, deepgram-sdk) and pydub."
        )

    # Determine file extension and optionally convert to WAV for consistency
    is_wav = (content_type and "wav" in content_type.lower()) or (
        len(audio_data) > 12 and audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE"
    )
    suffix = ".wav"
    data_to_use = audio_data

    if not is_wav:
        if _HAS_PYDUB:
            try:
                fmt = "webm" if (content_type and "webm" in content_type.lower()) else "mp3" if (content_type and "mp3" in content_type.lower()) else "ogg"
                seg = AudioSegment.from_file(io.BytesIO(audio_data), format=fmt)
                buf = io.BytesIO()
                seg = seg.set_channels(1)
                seg = seg.set_frame_rate(16000)
                seg.export(buf, format="wav")
                data_to_use = buf.getvalue()
                suffix = ".wav"
            except Exception as e:
                suffix = ".webm" if (content_type and "webm" in content_type.lower()) else ".mp3"
                data_to_use = audio_data
                print(f"[STT] webm/mp3->wav conversion failed: {e}; passing raw {suffix} ({len(audio_data)} bytes)")
        else:
            suffix = ".webm" if (content_type and "webm" in content_type.lower()) else ".mp3"
            data_to_use = audio_data

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = f.name
        try:
            f.write(data_to_use)
            f.flush()
        except Exception:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
            return ""

    try:
        model = _get_speech_model()
        if model is None:
            print("[STT] speech model not available (_get_speech_model returned None)")
            return ""
        text = model.transcribe_audio_file(path)
        out = (text or "").strip()
        if not out:
            print(f"[STT] model returned empty transcript (path={path!r}, content_type={content_type})")
        return out
    except Exception as e:
        print(f"[STT] transcription failed: {e}")
        return ""
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
