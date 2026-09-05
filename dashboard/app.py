from __future__ import annotations

import io
import math
import hashlib
import os
import shutil
import subprocess
import tempfile
import sys
import re
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# Streamlit renders indented Markdown as a code block.
# The AERIS UI contains intentionally indented multiline HTML, so
# normalize only HTML-bearing Markdown before Streamlit renders it.
_AERIS_ORIGINAL_MARKDOWN = st.markdown
_AERIS_HTML_RE = re.compile(
    r"<\s*/?\s*(?:div|style|span|b|strong|p|h[1-6]|table|thead|tbody|tr|td|th|section|details|summary|hr|br|svg|path|button)\b",
    re.IGNORECASE,
)


def _aeris_markdown(body, *args, **kwargs):
    """Render AERIS HTML through Streamlit's HTML renderer when available.

    Using st.html avoids Markdown's four-space code-block parsing entirely.
    Non-HTML Markdown continues through the normal Streamlit Markdown renderer.
    """
    if isinstance(body, str) and _AERIS_HTML_RE.search(body):
        html = textwrap.dedent(body).strip()
        html_renderer = getattr(st, "html", None)
        if callable(html_renderer):
            # st.html is the reliable path for raw HTML/CSS and cannot turn
            # indented HTML into a Markdown code block.
            return html_renderer(html)
        kwargs.setdefault("unsafe_allow_html", True)
        body = html
    return _AERIS_ORIGINAL_MARKDOWN(body, *args, **kwargs)


st.markdown = _aeris_markdown


# ============================================================
# AERIS
# Acoustic Intelligence & Airspace Operations Console
#
# UI:
#   Cinematic command center
#   Tabbed operator workflow
#   Radar / spectrogram / waveform / threat visuals
#
# AUDIO:
#   Up to 500 MB upload
#   Long recordings automatically chunked
#   Feature extraction performed chunk-by-chunk
#   Aggregated result passed into analysis/model layer
#
# IMPORTANT:
# The existing repository/model remains the source of truth.
# The heuristic analysis below is the current demonstration
# fallback and can be replaced by the repository inference call.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "AERIS"
APP_VERSION = "3.1"
TARGET_SR = 16_000

# Long recordings are divided into manageable windows.
CHUNK_SECONDS = 4
CHUNK_OVERLAP_SECONDS = 2

# UI upload target.
MAX_UPLOAD_MB = 500

# Number of waveform points displayed.
MAX_WAVEFORM_POINTS = 2200

# Number of chunks retained for detailed visualisation.
MAX_VISUAL_CHUNKS = 120


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    import librosa

    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

try:
    import soundfile as sf

    SOUNDFILE_OK = True
except Exception:
    SOUNDFILE_OK = False


# ============================================================
# REPOSITORY / MODEL INTEGRATION
# ============================================================

# dashboard/app.py lives one directory below the repository root.
# Keeping this path explicit lets the dashboard use the same model and
# feature-extraction modules as the training pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TORCH_OK = False
REPOSITORY_FEATURES_OK = False

try:
    import torch
    from models.acoustic_drone_model import AcousticDroneModel
    TORCH_OK = True
except Exception:
    torch = None
    AcousticDroneModel = None

try:
    from feature_extraction.mfcc import extract_mfcc
    from feature_extraction.mel import extract_mel_spectrogram
    from feature_extraction.spectral import extract_spectral_features
    from feature_extraction.chroma import extract_chroma
    from feature_extraction.zcr import extract_zcr
    from feature_extraction.energy import extract_energy
    from utils.audio_processor import normalize_audio, remove_dc_offset
    REPOSITORY_FEATURES_OK = True
except Exception:
    REPOSITORY_FEATURES_OK = False


MODEL_FEATURE_NAMES = (
    "mfcc",
    "mel",
    "spectral",
    "chroma",
    "zcr",
    "energy",
)

MODEL_CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "last_model.pt",
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AERIS | Acoustic Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FUTURISTIC THEME
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #03070c;
    --bg2: #071019;
    --panel: rgba(8,16,25,.94);
    --panel2: rgba(11,23,34,.92);
    --panel3: rgba(13,28,42,.88);

    --cyan: #29f2e2;
    --cyan2: #00bfcf;
    --blue: #54a9ff;
    --violet: #8e72ff;

    --green: #50e59b;
    --amber: #ffc45e;
    --red: #ff405b;

    --text: #edf7ff;
    --muted: #73879b;
    --muted2: #506273;

    --line: rgba(117,171,207,.14);
    --line-cyan: rgba(41,242,226,.27);
}

html,
body,
[class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    min-height: 100vh;

    background:
        radial-gradient(
            900px 500px at 82% -10%,
            rgba(73,55,220,.20),
            transparent 62%
        ),
        radial-gradient(
            800px 500px at 10% 0%,
            rgba(0,220,210,.09),
            transparent 60%
        ),
        radial-gradient(
            600px 400px at 50% 100%,
            rgba(0,130,255,.055),
            transparent 65%
        ),
        linear-gradient(
            180deg,
            #02060a 0%,
            #050a10 48%,
            #02060a 100%
        );

    color: var(--text);
}

.block-container {
    max-width: 1850px;
    padding: 1rem 1.35rem 2.5rem;
}

#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* =========================================================
   TOP BAR
   ========================================================= */

.aeris-top {
    display: flex;
    align-items: center;
    justify-content: space-between;

    border-bottom: 1px solid var(--line);

    padding: 4px 3px 13px;
    margin-bottom: 12px;
}

.aeris-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.aeris-symbol {
    width: 39px;
    height: 39px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 11px;

    border: 1px solid rgba(41,242,226,.34);

    background:
        linear-gradient(
            135deg,
            rgba(41,242,226,.14),
            rgba(142,114,255,.24)
        );

    color: var(--cyan);

    font-size: 20px;
    font-weight: 800;

    box-shadow:
        0 0 30px rgba(41,242,226,.08),
        inset 0 0 20px rgba(41,242,226,.035);
}

.aeris-name {
    font-family: "Space Grotesk";
    font-size: 19px;
    font-weight: 700;
    letter-spacing: .03em;
}

.aeris-desc {
    color: var(--muted);
    font-size: 8px;
    letter-spacing: .17em;
    text-transform: uppercase;
    margin-top: 2px;
}

.top-status {
    display: flex;
    gap: 7px;
    align-items: center;
}

.status-pill {
    border: 1px solid var(--line);
    border-radius: 999px;

    background: rgba(255,255,255,.025);

    padding: 6px 10px;

    color: #a8bacb;

    font-size: 9px;
    font-weight: 700;
    letter-spacing: .03em;
}

.status-pill.live {
    color: var(--green);
    border-color: rgba(80,229,155,.23);
    background: rgba(80,229,155,.045);
}

.status-dot {
    width: 6px;
    height: 6px;
    display: inline-block;

    margin-right: 5px;

    border-radius: 50%;
    background: currentColor;

    box-shadow: 0 0 11px currentColor;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;

    border: 1px solid var(--line);
    border-radius: 18px;

    padding: 22px 24px;

    background:
        linear-gradient(
            135deg,
            rgba(12,25,37,.96),
            rgba(8,12,23,.96)
        );

    box-shadow:
        0 20px 60px rgba(0,0,0,.26),
        inset 0 1px 0 rgba(255,255,255,.025);
}

.hero:after {
    content: "";

    position: absolute;
    right: -100px;
    top: -100px;

    width: 300px;
    height: 300px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(41,242,226,.10),
            transparent 68%
        );

    pointer-events: none;
}

.hero-kicker {
    color: var(--cyan);

    font-size: 8px;
    font-weight: 800;

    letter-spacing: .18em;
    text-transform: uppercase;
}

.hero h1 {
    font-family: "Space Grotesk";

    font-size: 31px;
    line-height: 1.1;

    letter-spacing: -.045em;

    margin: 5px 0 7px;
}

.hero p {
    max-width: 850px;

    color: var(--muted);

    font-size: 11px;
    line-height: 1.6;

    margin: 0;
}


/* =========================================================
   METRICS
   ========================================================= */

.metric {
    min-height: 88px;

    padding: 13px 14px;

    border-radius: 13px;
    border: 1px solid var(--line);

    background:
        linear-gradient(
            180deg,
            rgba(13,25,37,.97),
            rgba(7,13,20,.97)
        );

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.025);
}

.metric-label {
    color: #6e8295;

    font-size: 8px;
    font-weight: 800;

    letter-spacing: .10em;
    text-transform: uppercase;
}

.metric-value {
    font-family: "Space Grotesk";

    font-size: 23px;
    font-weight: 700;

    letter-spacing: -.035em;

    margin-top: 7px;
}

.metric-sub {
    color: var(--green);

    font-size: 8px;

    margin-top: 4px;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    border: 1px solid var(--line);
    border-radius: 14px;

    background:
        linear-gradient(
            180deg,
            rgba(10,19,29,.95),
            rgba(5,11,18,.95)
        );

    padding: 15px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.018);
}

.card-title {
    font-family: "Space Grotesk";

    font-size: 12px;
    font-weight: 700;

    letter-spacing: .01em;
}

.card-sub {
    color: var(--muted);

    font-size: 8px;

    margin-top: 3px;
}

.card-rule {
    border-top: 1px solid var(--line);
    margin: 10px 0;
}

.section-title {
    font-family: "Space Grotesk";

    font-size: 14px;
    font-weight: 700;

    margin: 17px 0 8px;
}

.eyebrow {
    color: var(--cyan);

    font-size: 8px;
    font-weight: 800;

    letter-spacing: .17em;
    text-transform: uppercase;
}


/* =========================================================
   CONTACTS
   ========================================================= */

.contact {
    border: 1px solid var(--line);
    border-radius: 11px;

    background: rgba(255,255,255,.015);

    padding: 10px 11px;
    margin-bottom: 7px;

    transition: .18s ease;
}

.contact:hover {
    border-color: var(--line-cyan);

    background:
        linear-gradient(
            90deg,
            rgba(41,242,226,.045),
            rgba(255,255,255,.015)
        );
}

.contact-id {
    font-size: 10px;
    font-weight: 700;
}

.contact-meta {
    color: var(--muted);

    font-size: 8px;

    margin-top: 4px;
}

.progress {
    height: 4px;

    overflow: hidden;

    border-radius: 99px;

    background: #142230;

    margin-top: 7px;
}

.progress > div {
    height: 100%;

    border-radius: 99px;

    background:
        linear-gradient(
            90deg,
            var(--cyan),
            var(--violet)
        );
}


/* =========================================================
   ALERTS
   ========================================================= */

.alert {
    border: 1px solid rgba(255,64,91,.22);

    border-radius: 11px;

    background: rgba(255,64,91,.055);

    padding: 10px 11px;

    margin: 7px 0;
}

.alert-title {
    color: #ff7285;

    font-size: 9px;
    font-weight: 800;
}

.alert-text {
    color: var(--muted);

    font-size: 8px;

    line-height: 1.5;

    margin-top: 3px;
}

.good {
    border: 1px solid rgba(80,229,155,.20);

    border-radius: 11px;

    background: rgba(80,229,155,.045);

    padding: 10px 11px;
}


/* =========================================================
   BIG NUMBERS
   ========================================================= */

.big-number {
    font-family: "Space Grotesk";

    font-size: 45px;
    font-weight: 700;

    letter-spacing: -.06em;
}


/* =========================================================
   AUDIO INGESTION
   ========================================================= */

.ingest {
    border: 1px solid rgba(41,242,226,.18);

    border-radius: 15px;

    padding: 16px;

    background:
        radial-gradient(
            circle at 80% 20%,
            rgba(41,242,226,.055),
            transparent 40%
        ),
        rgba(6,14,22,.92);
}

.ingest-title {
    font-family: "Space Grotesk";

    font-size: 15px;
    font-weight: 700;
}

.ingest-sub {
    color: var(--muted);

    font-size: 9px;

    line-height: 1.5;

    margin-top: 3px;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {
    height: 40px;

    color: #687c8e;

    font-size: 9px;
    font-weight: 800;

    letter-spacing: .08em;

    text-transform: uppercase;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--cyan);
}

div[data-baseweb="tab-highlight"] {
    background: var(--cyan);
    height: 2px;
}


/* =========================================================
   STREAMLIT CONTROLS
   ========================================================= */

.stButton > button,
.stDownloadButton > button {
    min-height: 36px;

    border-radius: 9px;

    border: 1px solid var(--line);

    background:
        linear-gradient(
            180deg,
            #111e2c,
            #0a131e
        );

    color: #eaf5fc;

    font-weight: 700;

    font-size: 10px;
}

.stButton > button:hover {
    border-color: rgba(41,242,226,.42);
    color: var(--cyan);

    box-shadow:
        0 0 20px rgba(41,242,226,.06);
}

div[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(
            135deg,
            rgba(8,21,31,.85),
            rgba(7,12,20,.85)
        );

    border: 1px dashed rgba(41,242,226,.28);

    border-radius: 13px;
}

.stFileUploader label,
.stSelectbox label,
.stTextInput label,
.stSlider label {
    color: #7e91a4 !important;

    font-size: 8px !important;
    font-weight: 700 !important;

    letter-spacing: .06em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(8,15,23,.85);
}

hr {
    border-color: var(--line);
}


/* =========================================================
   SYSTEM STRIP
   ========================================================= */

.system-strip {
    display: grid;

    grid-template-columns: repeat(5, 1fr);

    gap: 7px;

    margin-top: 9px;
}

.system-item {
    border: 1px solid var(--line);

    border-radius: 9px;

    background: rgba(255,255,255,.015);

    padding: 8px 9px;
}

.system-item b {
    font-size: 8px;
}

.system-item span {
    display: block;

    color: var(--muted);

    font-size: 7px;

    margin-top: 3px;
}


/* =========================================================
   INGESTION PIPELINE
   ========================================================= */

.pipeline {
    display: grid;

    grid-template-columns:
        repeat(5, 1fr);

    gap: 5px;

    margin-top: 11px;
}

.pipeline-item {
    position: relative;

    border: 1px solid var(--line);

    border-radius: 8px;

    padding: 8px;

    background: rgba(255,255,255,.014);

    text-align: center;
}

.pipeline-item.active {
    border-color: rgba(41,242,226,.26);

    box-shadow:
        0 0 22px rgba(41,242,226,.045);
}

.pipeline-item b {
    font-size: 8px;
}

.pipeline-item span {
    display: block;

    color: var(--muted);

    font-size: 7px;

    margin-top: 3px;
}


/* =========================================================
   SCROLL / TABLE
   ========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media(max-width: 900px) {

    .block-container {
        padding: .7rem;
    }

    .top-status {
        display: none;
    }

    .hero h1 {
        font-size: 25px;
    }

    .system-strip,
    .pipeline {
        grid-template-columns: repeat(2, 1fr);
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def seeded_rng(blob: bytes | None = None):
    seed = (
        42
        if not blob
        else int(hashlib.sha256(blob).hexdigest()[:8], 16)
    )

    return np.random.default_rng(seed)


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"

        value /= 1024

    return f"{value:.1f} TB"


def format_duration(seconds: float) -> str:
    seconds = max(0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


# ============================================================
# AUDIO DECODING
# ============================================================

def decode_with_ffmpeg(data: bytes, target_sr: int = TARGET_SR):
    """
    Decode arbitrary supported audio into mono float32 PCM.

    This is the preferred fallback for:
        MP3
        M4A
        OGG
        other formats supported by installed ffmpeg

    ffmpeg must be available on PATH.
    """

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return None, None

    try:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate(
            input=data,
            timeout=300,
        )

        if process.returncode != 0:
            return None, None

        y = np.frombuffer(
            stdout,
            dtype=np.float32,
        ).copy()

        return y, target_sr

    except Exception:
        return None, None


def decode_audio(data: bytes):
    """
    Decode uploaded audio.

    The function keeps compatibility with the original
    AERIS ingestion behavior while adding ffmpeg support.
    """

    if not data:
        return None, TARGET_SR

    # First try soundfile for formats it natively supports.
    if SOUNDFILE_OK:

        try:
            y, sr = sf.read(
                io.BytesIO(data),
                always_2d=False,
            )

            y = np.asarray(
                y,
                dtype=np.float32,
            )

            if y.ndim > 1:
                y = y.mean(axis=1)

            if sr != TARGET_SR and LIBROSA_OK:

                y = librosa.resample(
                    y,
                    orig_sr=sr,
                    target_sr=TARGET_SR,
                )

                sr = TARGET_SR

            return y.astype(np.float32), int(sr)

        except Exception:
            pass

    # Then try ffmpeg.
    y, sr = decode_with_ffmpeg(
        data,
        TARGET_SR,
    )

    if y is not None:
        return y, sr

    # Final librosa fallback.
    if LIBROSA_OK:

        try:
            y, sr = librosa.load(
                io.BytesIO(data),
                sr=TARGET_SR,
                mono=True,
            )

            return (
                y.astype(np.float32),
                int(sr),
            )

        except Exception:
            pass

    return None, TARGET_SR


# ============================================================
# REPOSITORY MODEL ADAPTER
# ============================================================

def discover_model_checkpoint() -> Path | None:
    """
    Find the repository's trained checkpoint without hard-coding
    a machine-specific absolute path.
    """
    for candidate in MODEL_CHECKPOINT_CANDIDATES:
        if candidate.is_file():
            return candidate

    # Fallback: allow a differently named .pt checkpoint under
    # models/checkpoints while still preferring the canonical names.
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    if checkpoint_dir.is_dir():
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        if candidates:
            return candidates[0]

    return None


@st.cache_resource(show_spinner=False)
def load_repository_model(checkpoint_key: str | None):
    """Load the repository detector and its stored validation threshold.

    ``checkpoint_key`` contains the path plus a file fingerprint so Streamlit
    reloads the model after a new best checkpoint is written during training.
    The dashboard never changes model weights.

    Returns ``(model, status_text, decision_threshold)``.
    """
    if not TORCH_OK:
        return None, "PYTORCH UNAVAILABLE", None

    if not checkpoint_key:
        return None, "CHECKPOINT NOT FOUND · HEURISTIC FALLBACK", None

    checkpoint_path = Path(checkpoint_key.split("::", 1)[0])

    try:
        # Match the inference/evaluation pipeline: use the model's canonical
        # constructor rather than duplicating architecture parameters here.
        model = AcousticDroneModel()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )
        if not isinstance(state_dict, dict):
            raise KeyError("Checkpoint does not contain model_state_dict/state_dict.")

        # Accept DataParallel checkpoints as well.
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model.eval()

        threshold = float(checkpoint.get("decision_threshold", 0.50))
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid decision threshold: {threshold}")

        return (
            model,
            f"REPOSITORY MODEL · {checkpoint_path.name}",
            threshold,
        )

    except Exception as exc:
        return None, f"MODEL LOAD FAILED · {type(exc).__name__}", None


def prepare_repository_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray] | None:
    """
    Run the same feature-family extractors used by the repository.

    Default output channels match AcousticDroneModel:
        MFCC     = 120
        Mel      = 128
        Spectral = 12
        Chroma   = 12
        ZCR      = 1
        Energy   = 1
    """
    if not REPOSITORY_FEATURES_OK:
        return None

    audio = np.asarray(waveform, dtype=np.float32)

    try:
        audio = remove_dc_offset(audio)
        audio = normalize_audio(audio, 0.99)

        features = {
            "mfcc": extract_mfcc(audio, sample_rate),
            "mel": extract_mel_spectrogram(audio, sample_rate),
            "spectral": extract_spectral_features(audio, sample_rate),
            "chroma": extract_chroma(audio, sample_rate),
            "zcr": extract_zcr(audio, sample_rate),
            "energy": extract_energy(audio, sample_rate),
        }

        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in features.items()
        }

    except Exception:
        return None


def repository_model_predict(
    model,
    features: dict[str, np.ndarray] | None,
) -> tuple[float | None, np.ndarray | None]:
    """
    Run one model-compatible inference window.

    Returns:
        drone_probability, feature_attention
    """
    if model is None or features is None or not TORCH_OK:
        return None, None

    try:
        batch = {
            name: torch.from_numpy(
                np.ascontiguousarray(features[name], dtype=np.float32)
            ).unsqueeze(0)
            for name in MODEL_FEATURE_NAMES
        }

        with torch.inference_mode():
            logits, attention = model(batch)
            probability = torch.softmax(logits, dim=1)[0, 1].item()

        return float(probability), attention[0].detach().cpu().numpy()

    except Exception:
        return None, None


# ============================================================
# CHUNKING
# ============================================================

def chunk_audio(
    y: np.ndarray,
    sr: int,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
):
    """
    Yield overlapping audio windows.

    Long recordings therefore never need to be processed as
    one giant feature-extraction window.
    """

    if y is None or len(y) == 0:
        return

    chunk_samples = int(
        chunk_seconds * sr
    )

    overlap_samples = int(
        overlap_seconds * sr
    )

    step = max(
        1,
        chunk_samples - overlap_samples,
    )

    total = len(y)

    start = 0
    chunk_index = 0

    while start < total:

        end = min(
            start + chunk_samples,
            total,
        )

        chunk = y[start:end]

        if len(chunk) >= max(
            int(sr * 1.0),
            64,
        ):

            yield {
                "index": chunk_index,
                "start": start / sr,
                "end": end / sr,
                "audio": chunk,
            }

        if end >= total:
            break

        start += step
        chunk_index += 1


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_chunk_features(
    y: np.ndarray,
    sr: int,
):
    """
    Extract the same broad feature families used by the
    original application, but on one manageable chunk.
    """

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    rms = float(
        np.sqrt(
            np.mean(y * y) + 1e-12
        )
    )

    peak = float(
        np.max(np.abs(y)) + 1e-9
    )

    repository_features = prepare_repository_features(y, sr)

    if LIBROSA_OK:

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr,
                )
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y
                )
            )
        )

        if repository_features is not None:
            mfcc = repository_features["mfcc"]
            mel = repository_features["mel"]
            db = mel
        else:
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=20,
            )

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=64,
            )

            db = librosa.power_to_db(
                mel,
                ref=np.max,
            )

        onset = float(
            np.mean(
                librosa.onset.onset_strength(
                    y=y,
                    sr=sr,
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sr,
                    roll_percent=.85,
                )
            )
        )

    else:

        n = min(
            len(y),
            sr * 8,
        )

        segment = (
            y[:n]
            * np.hanning(n)
        )

        spec = np.abs(
            np.fft.rfft(segment)
        )

        freqs = np.fft.rfftfreq(
            n,
            1 / sr,
        )

        centroid = float(
            (spec * freqs).sum()
            / (spec.sum() + 1e-9)
        )

        zcr = float(
            np.mean(
                np.diff(
                    np.signbit(
                        y[:n]
                    )
                )
            )
        )

        mfcc = np.zeros(
            (20, 64)
        )

        mel = np.maximum(
            spec[:2048, None],
            1e-8,
        )

        db = 20 * np.log10(
            mel / mel.max()
        )

        onset = float(
            np.std(spec)
        )

        rolloff = float(
            centroid * 1.7
        )

    return {
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "zcr": zcr,
        "mfcc": mfcc,
        "mel": mel,
        "db": db,
        "onset": onset,
        "rolloff": rolloff,
        "model_features": repository_features,
    }


# ============================================================
# LONG RECORDING ANALYSIS
# ============================================================


def _read_f32_samples(stream, sample_count: int) -> np.ndarray:
    """Read up to sample_count float32 PCM samples from an ffmpeg pipe."""
    target_bytes = sample_count * 4
    chunks = []
    received = 0

    while received < target_bytes:
        piece = stream.read(target_bytes - received)
        if not piece:
            break
        chunks.append(piece)
        received += len(piece)

    if not chunks:
        return np.empty(0, dtype=np.float32)

    raw = b"".join(chunks)
    usable = len(raw) - (len(raw) % 4)
    return np.frombuffer(raw[:usable], dtype=np.float32).copy()


def iter_streamed_audio_chunks(
    data: bytes,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
):
    """
    Decode compressed/uploaded audio through ffmpeg and yield fixed windows
    without materialising the complete decoded recording in RAM.

    This is the preferred path for large recordings. It keeps memory roughly
    proportional to one model window rather than the complete audio duration.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not data:
        return

    temp_path = None
    process = None

    chunk_samples = int(round(chunk_seconds * TARGET_SR))
    overlap_samples = int(round(overlap_seconds * TARGET_SR))
    step_samples = max(1, chunk_samples - overlap_samples)

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".audio",
            delete=False,
        ) as temp:
            temp.write(data)
            temp.flush()
            temp_path = temp.name

        process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                temp_path,
                "-ac",
                "1",
                "-ar",
                str(TARGET_SR),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        first = True
        previous_tail = np.empty(0, dtype=np.float32)
        index = 0

        while True:
            if first:
                new_samples = _read_f32_samples(
                    process.stdout,
                    chunk_samples,
                )
                current = new_samples
                first = False
            else:
                new_samples = _read_f32_samples(
                    process.stdout,
                    step_samples,
                )

                if new_samples.size == 0:
                    break

                current = np.concatenate(
                    [previous_tail, new_samples]
                )

            if current.size < max(
                int(TARGET_SR * 1.0),
                64,
            ):
                break

            start_seconds = index * (
                step_samples / TARGET_SR
            )

            end_seconds = start_seconds + (
                current.size / TARGET_SR
            )

            yield {
                "index": index,
                "start": start_seconds,
                "end": end_seconds,
                "audio": current,
            }

            previous_tail = (
                current[-overlap_samples:].copy()
                if overlap_samples > 0
                else np.empty(0, dtype=np.float32)
            )

            index += 1

            if new_samples.size < step_samples:
                break

    finally:
        if process is not None:
            try:
                process.stdout.close()
            except Exception:
                pass

            try:
                process.stderr.close()
            except Exception:
                pass

            try:
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def analyze_audio_chunks(
    data: bytes | None,
    progress_callback=None,
):
    """
    Analyze an uploaded recording using model-compatible fixed windows.

    Large files are decoded through an ffmpeg stream so the complete
    decoded PCM does not have to coexist in memory with the upload.

    Windowing matches the repository preprocessing pipeline:
        4.0 second window
        2.0 second hop

    The existing repository feature extractors and AcousticDroneModel are
    used when the trained checkpoint is available. The heuristic layer
    remains the transparent fallback.
    """
    rng = seeded_rng(data)

    raw_data = data or b""

    # --------------------------------------------------------
    # Model / feature engine
    # --------------------------------------------------------
    checkpoint_path = discover_model_checkpoint()

    checkpoint_key = None
    if checkpoint_path:
        try:
            stat = checkpoint_path.stat()
            checkpoint_key = f"{checkpoint_path}::{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            checkpoint_key = str(checkpoint_path)

    repository_model, model_status, decision_threshold = load_repository_model(
        checkpoint_key
    )

    model_probabilities: list[float] = []
    attention_rows: list[np.ndarray] = []

    chunks = []
    visual_store = []
    waveform_store = []

    total_duration = 0.0
    last_end = 0.0
    used_streaming = False

    # --------------------------------------------------------
    # Preferred path: streaming ffmpeg decode
    # --------------------------------------------------------
    stream = iter_streamed_audio_chunks(
        raw_data,
        CHUNK_SECONDS,
        CHUNK_OVERLAP_SECONDS,
    )

    if stream is not None:
        used_streaming = True

        try:
            for chunk in stream:
                features = extract_chunk_features(
                    chunk["audio"],
                    TARGET_SR,
                )

                model_features = features.get(
                    "model_features"
                )

                model_probability, attention = (
                    repository_model_predict(
                        repository_model,
                        model_features,
                    )
                )

                if model_probability is not None:
                    model_probabilities.append(
                        model_probability
                    )

                    if attention is not None:
                        attention_rows.append(
                            attention
                        )

                features["model_probability"] = (
                    model_probability
                )

                chunks.append(
                    {
                        "index": chunk["index"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "rms": features["rms"],
                        "peak": features["peak"],
                        "centroid": features["centroid"],
                        "zcr": features["zcr"],
                        "onset": features["onset"],
                        "rolloff": features["rolloff"],
                        "model_probability": model_probability,
                        "model_decision": (
                            bool(model_probability >= decision_threshold)
                            if model_probability is not None and decision_threshold is not None
                            else None
                        ),
                    }
                )

                # Keep only a bounded number of feature matrices for
                # the spectrogram / MFCC visualisations.
                visual_store.append(
                    (
                        chunk["index"],
                        features["db"],
                        features["mfcc"],
                        features["rms"],
                    )
                )

                if len(visual_store) > (
                    MAX_VISUAL_CHUNKS * 2
                ):
                    visual_store = visual_store[
                        ::2
                    ]

                # Store a small waveform sketch per window.
                display_points = min(
                    16,
                    len(chunk["audio"]),
                )

                if display_points > 1:
                    sampled = np.linspace(
                        0,
                        len(chunk["audio"]) - 1,
                        display_points,
                    ).astype(int)

                    waveform_store.extend(
                        [
                            (
                                chunk["start"]
                                + idx / TARGET_SR,
                                float(chunk["audio"][idx]),
                            )
                            for idx in sampled
                        ]
                    )

                total_duration = max(
                    total_duration,
                    chunk["end"],
                )
                last_end = chunk["end"]

                if progress_callback:
                    progress_callback(
                        len(chunks)
                    )

        except Exception:
            # If ffmpeg streaming cannot decode this particular input,
            # fall back to the original decoder below.
            used_streaming = False
            chunks.clear()
            visual_store.clear()
            waveform_store.clear()
            model_probabilities.clear()
            attention_rows.clear()
            total_duration = 0.0
            last_end = 0.0

    # --------------------------------------------------------
    # Fallback path: existing in-memory decoder
    # --------------------------------------------------------
    if not chunks:
        y, sr = decode_audio(
            raw_data
        )

        if y is None or len(y) < 64:
            duration = 10.0

            t = np.linspace(
                0,
                duration,
                int(TARGET_SR * duration),
                endpoint=False,
            )

            y = (
                .30 * np.sin(
                    2 * np.pi * 235 * t
                )
                + .11 * np.sin(
                    2 * np.pi * 470 * t
                )
                + .055 * np.sin(
                    2 * np.pi * 705 * t
                )
                + .035 * rng.normal(
                    size=t.size
                )
            ).astype(np.float32)

            sr = TARGET_SR
            source = "DEMO SIGNAL"

        else:
            source = "UPLOADED AUDIO"

        duration = len(y) / max(sr, 1)
        total_duration = duration

        for chunk in chunk_audio(
            y,
            sr,
            CHUNK_SECONDS,
            CHUNK_OVERLAP_SECONDS,
        ):
            features = extract_chunk_features(
                chunk["audio"],
                sr,
            )

            model_probability, attention = (
                repository_model_predict(
                    repository_model,
                    features.get("model_features"),
                )
            )

            if model_probability is not None:
                model_probabilities.append(
                    model_probability
                )

                if attention is not None:
                    attention_rows.append(
                        attention
                    )

            chunks.append(
                {
                    "index": chunk["index"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "rms": features["rms"],
                    "peak": features["peak"],
                    "centroid": features["centroid"],
                    "zcr": features["zcr"],
                    "onset": features["onset"],
                    "rolloff": features["rolloff"],
                    "model_probability": model_probability,
                }
            )

            visual_store.append(
                (
                    chunk["index"],
                    features["db"],
                    features["mfcc"],
                    features["rms"],
                )
            )

            if len(visual_store) > (
                MAX_VISUAL_CHUNKS * 2
            ):
                visual_store = visual_store[
                    ::2
                ]

            display_points = min(
                16,
                len(chunk["audio"]),
            )

            if display_points > 1:
                sampled = np.linspace(
                    0,
                    len(chunk["audio"]) - 1,
                    display_points,
                ).astype(int)

                waveform_store.extend(
                    [
                        (
                            chunk["start"]
                            + idx / sr,
                            float(chunk["audio"][idx]),
                        )
                        for idx in sampled
                    ]
                )

            if progress_callback:
                progress_callback(
                    len(chunks)
                )

    if not chunks:
        return analyze_audio_legacy(
            data
        )

    # --------------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------------
    rms_values = np.array(
        [c["rms"] for c in chunks],
        dtype=np.float32,
    )

    peak_values = np.array(
        [c["peak"] for c in chunks],
        dtype=np.float32,
    )

    centroid_values = np.array(
        [c["centroid"] for c in chunks],
        dtype=np.float32,
    )

    zcr_values = np.array(
        [c["zcr"] for c in chunks],
        dtype=np.float32,
    )

    onset_values = np.array(
        [c["onset"] for c in chunks],
        dtype=np.float32,
    )

    rolloff_values = np.array(
        [c["rolloff"] for c in chunks],
        dtype=np.float32,
    )

    rms = float(np.mean(rms_values))
    peak = float(np.max(peak_values))
    centroid = float(np.mean(centroid_values))
    zcr = float(np.mean(zcr_values))
    onset = float(np.mean(onset_values))
    rolloff = float(np.mean(rolloff_values))

    # --------------------------------------------------------
    # Bounded visual feature set
    # --------------------------------------------------------
    visual_store = sorted(
        visual_store,
        key=lambda item: item[0],
    )

    if len(visual_store) > MAX_VISUAL_CHUNKS:
        positions = np.linspace(
            0,
            len(visual_store) - 1,
            MAX_VISUAL_CHUNKS,
        ).astype(int)

        visual_store = [
            visual_store[i]
            for i in positions
        ]

    mel_blocks = [
        item[1]
        for item in visual_store
    ]

    mfcc_blocks = [
        item[2]
        for item in visual_store
    ]

    if mel_blocks:
        min_cols = min(
            block.shape[1]
            for block in mel_blocks
        )

        mel = np.concatenate(
            [
                block[:, :min_cols]
                for block in mel_blocks
            ],
            axis=1,
        )
    else:
        mel = np.zeros(
            (128, 1),
            dtype=np.float32,
        )

    if mfcc_blocks:
        min_cols = min(
            block.shape[1]
            for block in mfcc_blocks
        )

        mfcc = np.concatenate(
            [
                block[:, :min_cols]
                for block in mfcc_blocks
            ],
            axis=1,
        )
    else:
        mfcc = np.zeros(
            (120, 1),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Existing heuristic recognition/scoring layer
    # --------------------------------------------------------
    likelihood = float(
        np.clip(
            .35
            + .25 * np.tanh(
                (centroid - 800) / 1300
            )
            + .18 * np.tanh(
                (rms - .025) * 14
            )
            + .06 * np.tanh(
                onset / 3
            ),
            .02,
            .98,
        )
    )

    category = [
        "Multirotor",
        "Fixed-wing",
        "Unknown",
    ][
        int(
            np.clip(
                (centroid - 500)
                / 1700
                * 2.1,
                0,
                2,
            )
        )
    ]

    size = [
        "Micro",
        "Small",
        "Medium",
        "Large",
    ][
        int(
            np.clip(
                (centroid - 300)
                / 1300
                * 3.6,
                0,
                3,
            )
        )
    ]

    movement = [
        "Hover",
        "Cruise",
        "Approach",
        "Departing",
    ][
        int(
            np.clip(
                (onset / 3) * 3.2,
                0,
                3,
            )
        )
    ]

    direction = int(
        (
            centroid * .19
            + onset * 23
        )
        % 360
    )

    distance = float(
        np.clip(
            8
            + (1 - likelihood) * 25
            + rng.normal(0, .7),
            2.5,
            45,
        )
    )

    speed = float(
        np.clip(
            2
            + likelihood * 18
            + (onset % 1) * 5,
            .5,
            28,
        )
    )

    count = int(
        np.clip(
            1
            + round(
                likelihood * 2
                + max(0, onset - 1) * .25
            ),
            1,
            4,
        )
    )

    threat = float(
        np.clip(
            likelihood * 62
            + (1 - distance / 50) * 18
            + min(speed / 28, 1) * 12
            + (count - 1) * 4,
            0,
            100,
        )
    )

    # Prefer the trained repository classifier when available.
    model_mean_probability = (
        float(np.mean(model_probabilities))
        if model_probabilities
        else None
    )

    model_peak_probability = (
        float(np.max(model_probabilities))
        if model_probabilities
        else None
    )

    model_window_decision = None
    model_positive_windows = 0
    if model_probabilities and decision_threshold is not None:
        model_positive_windows = int(
            sum(probability >= decision_threshold for probability in model_probabilities)
        )
        # Window-level decisions use the threshold selected on validation.
        # Recording-level aggregation is intentionally kept explicit: the
        # dashboard reports both mean probability and positive-window count
        # rather than inventing a new calibrated recording threshold.
        model_window_decision = bool(model_positive_windows > 0)

    if model_mean_probability is not None:
        # Do not blend model output with the old heuristic likelihood.
        likelihood = float(np.clip(model_mean_probability, 0.0, 1.0))

        threat = float(
            np.clip(
                likelihood * 62
                + (1 - distance / 50) * 18
                + min(speed / 28, 1) * 12
                + (count - 1) * 4,
                0,
                100,
            )
        )

    level = (
        "CRITICAL"
        if threat >= 82
        else "HIGH"
        if threat >= 64
        else "MEDIUM"
        if threat >= 40
        else "LOW"
    )

    # --------------------------------------------------------
    # Compact waveform representation
    # --------------------------------------------------------
    if waveform_store:
        waveform_store.sort(key=lambda item: item[0])

        waveform_x = np.asarray(
            [item[0] for item in waveform_store],
            dtype=np.float32,
        )

        waveform_y = np.asarray(
            [item[1] for item in waveform_store],
            dtype=np.float32,
        )

        if waveform_y.size > MAX_WAVEFORM_POINTS:
            positions = np.linspace(
                0,
                waveform_y.size - 1,
                MAX_WAVEFORM_POINTS,
            ).astype(int)

            waveform_x = waveform_x[positions]
            waveform_y = waveform_y[positions]
    else:
        waveform_x = np.array([0.0], dtype=np.float32)
        waveform_y = np.array([0.0], dtype=np.float32)

    return {
        "y": None,
        "waveform_y": waveform_y,
        "waveform_x": waveform_x,
        "sr": TARGET_SR,
        "duration": total_duration,
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "zcr": zcr,
        "mfcc": mfcc,
        "mel": mel,
        "db": mel,
        "onset": onset,
        "rolloff": rolloff,
        "source": "UPLOADED AUDIO" if raw_data else "DEMO SIGNAL",
        "drone_likelihood": likelihood,
        "category": category,
        "size": size,
        "movement": movement,
        "direction": direction,
        "distance": distance,
        "speed": speed,
        "count": count,
        "threat": threat,
        "level": level,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": CHUNK_OVERLAP_SECONDS,
        "model_status": model_status,
        "decision_threshold": decision_threshold,
        "model_mean_probability": model_mean_probability,
        "model_peak_probability": model_peak_probability,
        "model_positive_windows": model_positive_windows,
        "model_window_decision": model_window_decision,
        "model_attention": (
            np.mean(
                np.stack(attention_rows),
                axis=0,
            ).tolist()
            if attention_rows
            else None
        ),
        "streaming_decode": used_streaming,
    }


# ============================================================
# ORIGINAL ANALYSIS COMPATIBILITY WRAPPER
# ============================================================

def analyze_audio_legacy(data: bytes | None):
    """
    Compatibility fallback preserving the original AERIS
    analysis behavior for cases where chunk decoding fails.

    Existing fields are retained.
    """

    rng = seeded_rng(data)

    y, sr = decode_audio(
        data or b""
    )

    if y is None or len(y) < 64:

        duration = 10.0

        t = np.linspace(
            0,
            duration,
            int(16000 * duration),
            endpoint=False,
        )

        y = (
            .30 * np.sin(
                2 * np.pi * 235 * t
            )
            + .11 * np.sin(
                2 * np.pi * 470 * t
            )
            + .055 * np.sin(
                2 * np.pi * 705 * t
            )
            + .035 * rng.normal(
                size=t.size
            )
        ).astype(np.float32)

        sr = 16000
        source = "DEMO SIGNAL"

    else:

        source = "UPLOADED AUDIO"

    duration = len(y) / max(
        sr,
        1,
    )

    if LIBROSA_OK:

        features = extract_chunk_features(
            y[
                : min(
                    len(y),
                    sr * CHUNK_SECONDS,
                )
            ],
            sr,
        )

        centroid = features["centroid"]
        zcr = features["zcr"]
        mfcc = features["mfcc"]
        mel = features["mel"]
        db = features["db"]
        onset = features["onset"]
        rolloff = features["rolloff"]

    else:

        n = min(
            len(y),
            sr * 8,
        )

        spec = np.abs(
            np.fft.rfft(
                y[:n]
                * np.hanning(n)
            )
        )

        freqs = np.fft.rfftfreq(
            n,
            1 / sr,
        )

        centroid = float(
            (spec * freqs).sum()
            / (spec.sum() + 1e-9)
        )

        zcr = float(
            np.mean(
                np.diff(
                    np.signbit(
                        y[:n]
                    )
                )
            )
        )

        mfcc = np.zeros(
            (20, 64)
        )

        mel = np.maximum(
            spec[:2048, None],
            1e-8,
        )

        db = 20 * np.log10(
            mel / mel.max()
        )

        onset = float(
            np.std(spec)
        )

        rolloff = float(
            centroid * 1.7
        )

    rms = float(
        np.sqrt(
            np.mean(y * y)
            + 1e-12
        )
    )

    peak = float(
        np.max(
            np.abs(y)
        )
        + 1e-9
    )

    likelihood = float(
        np.clip(
            .35
            + .25 * np.tanh(
                (centroid - 800)
                / 1300
            )
            + .18 * np.tanh(
                (rms - .025) * 14
            )
            + .06 * np.tanh(
                onset / 3
            ),
            .02,
            .98,
        )
    )

    category = [
        "Multirotor",
        "Fixed-wing",
        "Unknown",
    ][
        int(
            np.clip(
                (centroid - 500)
                / 1700
                * 2.1,
                0,
                2,
            )
        )
    ]

    size = [
        "Micro",
        "Small",
        "Medium",
        "Large",
    ][
        int(
            np.clip(
                (centroid - 300)
                / 1300
                * 3.6,
                0,
                3,
            )
        )
    ]

    movement = [
        "Hover",
        "Cruise",
        "Approach",
        "Departing",
    ][
        int(
            np.clip(
                (onset / 3) * 3.2,
                0,
                3,
            )
        )
    ]

    direction = int(
        (
            centroid * .19
            + onset * 23
        )
        % 360
    )

    distance = float(
        np.clip(
            8
            + (1 - likelihood) * 25
            + rng.normal(0, .7),
            2.5,
            45,
        )
    )

    speed = float(
        np.clip(
            2
            + likelihood * 18
            + (onset % 1) * 5,
            .5,
            28,
        )
    )

    count = int(
        np.clip(
            1
            + round(
                likelihood * 2
                + max(0, onset - 1) * .25
            ),
            1,
            4,
        )
    )

    threat = float(
        np.clip(
            likelihood * 62
            + (1 - distance / 50) * 18
            + min(speed / 28, 1) * 12
            + (count - 1) * 4,
            0,
            100,
        )
    )

    level = (
        "CRITICAL"
        if threat >= 82
        else "HIGH"
        if threat >= 64
        else "MEDIUM"
        if threat >= 40
        else "LOW"
    )

    waveform_step = max(
        1,
        len(y) // MAX_WAVEFORM_POINTS,
    )

    waveform_y = y[
        ::waveform_step
    ]

    waveform_x = np.linspace(
        0,
        duration,
        len(waveform_y),
    )

    return {
        "y": y,
        "waveform_y": waveform_y,
        "waveform_x": waveform_x,
        "sr": sr,
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "zcr": zcr,
        "mfcc": mfcc,
        "mel": mel,
        "db": db,
        "onset": onset,
        "rolloff": rolloff,
        "source": source,
        "drone_likelihood": likelihood,
        "category": category,
        "size": size,
        "movement": movement,
        "direction": direction,
        "distance": distance,
        "speed": speed,
        "count": count,
        "threat": threat,
        "level": level,
        "chunks": [],
        "chunk_count": 1,
        "chunk_seconds": duration,
        "overlap_seconds": 0,
    }


def analyze_audio(data, progress_callback=None):
    """Public analysis API; long recordings use fixed model windows."""
    return analyze_audio_chunks(data, progress_callback=progress_callback)


# ============================================================
# VISUALIZATIONS
# ============================================================

_PLOTLY_SEQUENCE = 0


def aeris_plotly_chart(fig, *, width="stretch", config=None):
    """Render a Plotly figure with a deterministic per-run unique key."""
    global _PLOTLY_SEQUENCE
    _PLOTLY_SEQUENCE += 1
    st.plotly_chart(
        fig,
        width=width,
        config=config or {"displayModeBar": False},
        key=f"aeris_plot_{_PLOTLY_SEQUENCE}",
    )


def waveform_fig(a):

    if "waveform_y" in a:

        y = a["waveform_y"]
        x = a["waveform_x"]

    else:

        y = a["y"]

        step = max(
            1,
            len(y) // MAX_WAVEFORM_POINTS,
        )

        y = y[::step]

        x = np.linspace(
            0,
            a["duration"],
            len(y),
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                width=1.1
            ),
            hovertemplate=(
                "%{x:.2f}s"
                "<br>%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=25,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="TIME",
            showgrid=False,
        ),
        yaxis=dict(
            title="AMPLITUDE",
            gridcolor="rgba(255,255,255,.045)",
        ),
        font=dict(
            color="#8090a3",
            size=8,
        ),
    )

    return fig


def spectrogram_fig(a):

    db = a["db"]

    fig = go.Figure(
        go.Heatmap(
            z=db,
            colorscale=[
                [0.00, "#050812"],
                [0.18, "#10124a"],
                [0.38, "#25228b"],
                [0.60, "#007c9b"],
                [0.78, "#22d7bd"],
                [1.00, "#eaff73"],
            ],
            showscale=False,
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=25,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="TIME / CHUNKS",
            showgrid=False,
        ),
        yaxis=dict(
            title="MEL FREQUENCY",
            gridcolor="rgba(255,255,255,.035)",
        ),
        font=dict(
            color="#8090a3",
            size=8,
        ),
    )

    return fig


def radar_fig(a):

    theta = np.linspace(
        0,
        2 * np.pi,
        180,
    )

    base = (
        .22
        + .10
        * np.sin(
            theta * 3
            + .4
        )
    )

    contact_angle = math.radians(
        a["direction"]
    )

    angular_distance = np.angle(
        np.exp(
            1j
            * (
                theta
                - contact_angle
            )
        )
    )

    spread = np.exp(
        -(
            angular_distance**2
        )
        / (
            2
            * .17**2
        )
    )

    signal = (
        base
        + (
            .65
            + .18
            * a["drone_likelihood"]
        )
        * spread
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=signal,
            theta=np.degrees(theta),
            mode="lines",
            fill="toself",
            line=dict(
                width=1.5
            ),
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[1.02],
            theta=[a["direction"]],
            mode="markers+text",
            text=["CONTACT"],
            textposition="top center",
            marker=dict(
                size=10
            ),
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(
            l=12,
            r=12,
            t=20,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#8090a3",
            size=8,
        ),
        polar=dict(
            bgcolor="rgba(2,8,13,.72)",
            radialaxis=dict(
                showticklabels=False,
                gridcolor="rgba(255,255,255,.05)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,.05)",
                direction="clockwise",
            ),
        ),
        showlegend=False,
    )

    return fig


def threat_gauge(a):

    value = a["threat"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(
                font=dict(
                    size=39,
                    color="#edf7ff",
                )
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor="#506273",
                    tickfont=dict(
                        size=8
                    ),
                ),
                bar=dict(
                    thickness=.25
                ),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    {
                        "range": [0, 40],
                        "color": "rgba(80,229,155,.13)",
                    },
                    {
                        "range": [40, 64],
                        "color": "rgba(255,196,94,.13)",
                    },
                    {
                        "range": [64, 82],
                        "color": "rgba(255,128,70,.13)",
                    },
                    {
                        "range": [82, 100],
                        "color": "rgba(255,64,91,.16)",
                    },
                ],
            ),
            title=dict(
                text=f"<b>{a['level']}</b>",
                font=dict(
                    size=10
                ),
            ),
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=10,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#8090a3"
        ),
    )

    return fig


def trend_fig(history):

    if history.empty:
        return go.Figure()

    h = history.tail(30)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=h.timestamp,
            y=h.threat,
            name="THREAT",
            mode="lines+markers",
            line=dict(
                width=2
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=h.timestamp,
            y=h.confidence * 100,
            name="CONFIDENCE",
            mode="lines",
            line=dict(
                width=1.4,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        height=270,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=25,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            title="INDEX",
            gridcolor="rgba(255,255,255,.05)",
        ),
        legend=dict(
            orientation="h",
            y=1.12,
        ),
        font=dict(
            color="#8090a3",
            size=8,
        ),
    )

    return fig


# ============================================================
# HISTORY
# ============================================================

def make_history():

    rng = np.random.default_rng(7)

    n = 24

    ts = [
        datetime.now()
        - timedelta(
            minutes=(n - i) * 4
        )
        for i in range(n)
    ]

    confidence = np.clip(
        rng.normal(.82, .09, n),
        .52,
        .99,
    )

    threat = np.clip(
        confidence * 65
        + rng.normal(0, 12, n),
        8,
        98,
    )

    levels = [
        (
            "LOW"
            if x < 40
            else "MEDIUM"
            if x < 64
            else "HIGH"
            if x < 82
            else "CRITICAL"
        )
        for x in threat
    ]

    return pd.DataFrame(
        {
            "timestamp": ts,
            "source": [
                f"ARRAY-{i % 3 + 1:02d}"
                for i in range(n)
            ],
            "category": rng.choice(
                [
                    "Multirotor",
                    "Fixed-wing",
                    "Unknown",
                ],
                n,
                p=[.58, .27, .15],
            ),
            "distance_m": np.round(
                rng.uniform(
                    6,
                    42,
                    n,
                ),
                1,
            ),
            "speed_mps": np.round(
                rng.uniform(
                    2,
                    24,
                    n,
                ),
                1,
            ),
            "threat": np.round(
                threat,
                1,
            ),
            "confidence": np.round(
                confidence,
                3,
            ),
            "level": levels,
        }
    )


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = analyze_audio(None)

if "history" not in st.session_state:
    st.session_state.history = make_history()

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "uploaded_size" not in st.session_state:
    st.session_state.uploaded_size = 0


a = st.session_state.analysis


# ============================================================
# TOP BAR
# ============================================================

# NOTE: Streamlit's server upload ceiling is configured in
# .streamlit/config.toml. The app also enforces the same 500 MB ceiling
# after upload so the operator sees a clear error instead of a silent failure.

st.markdown(
    """
<div class="aeris-top">

    <div class="aeris-logo">

        <div class="aeris-symbol">
            ◈
        </div>

        <div>
            <div class="aeris-name">
                AERIS
            </div>

            <div class="aeris-desc">
                Acoustic Intelligence Console
            </div>
        </div>

    </div>

    <div class="top-status">

        <div class="status-pill">
            <span class="status-dot"
                  style="color:#54a9ff"></span>
            LOCAL NODE
        </div>

        <div class="status-pill live">
            <span class="status-dot"></span>
            PIPELINE NOMINAL
        </div>

        <div class="status-pill">
            16 KHZ / MONO
        </div>

        <div class="status-pill">
            500 MB INGEST
        </div>

    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
<div class="hero">

    <div class="hero-kicker">
        ACOUSTIC AIRSPACE MONITORING · COMMAND INTERFACE
    </div>

    <h1>
        Intelligent acoustic threat picture
    </h1>

    <p>
        Analyze long-form acoustic evidence, automatically segment
        recordings into inference windows, inspect signal intelligence,
        and surface operator-facing recognition and threat indicators
        from a single research console.
    </p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PRIMARY TABS
# ============================================================

tabs = st.tabs(
    [
        "⌁ COMMAND",
        "◉ AUDIO LAB",
        "◎ AIRSPACE",
        "▦ HISTORY",
        "⚠ THREAT INTEL",
        "◇ MODEL INSIGHTS",
    ]
)


# ============================================================
# TAB 1 — COMMAND
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">System posture</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    cards = [
        (
            "Drone likelihood",
            f"{a['drone_likelihood'] * 100:.0f}%",
            "signal confidence",
        ),
        (
            "Threat score",
            f"{a['threat']:.0f}/100",
            a["level"],
        ),
        (
            "Estimated range",
            f"{a['distance']:.1f} m",
            "acoustic estimate",
        ),
        (
            "Velocity",
            f"{a['speed']:.1f} m/s",
            a["movement"],
        ),
        (
            "Active contacts",
            str(a["count"]),
            "current window",
        ),
    ]

    for col, item in zip(
        cols,
        cards,
    ):

        with col:

            st.markdown(
                metric_card(
                    *item
                ),
                unsafe_allow_html=True,
            )

    left, center, right = st.columns(
        [1.05, 1.75, 1.05]
    )

    # --------------------------------------------------------
    # CONTACTS
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">Active contacts</div>',
            unsafe_allow_html=True,
        )

        for i in range(
            max(
                1,
                a["count"],
            )
        ):

            likelihood = max(
                .46,
                min(
                    .98,
                    a["drone_likelihood"]
                    - i * .11,
                ),
            )

            threat = max(
                18,
                min(
                    96,
                    a["threat"]
                    - i * 9,
                ),
            )

            st.markdown(
                f"""
                <div class="contact">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                    ">

                        <span class="contact-id">
                            CONTACT {i+1:02d}
                        </span>

                        <span style="
                            color:#8da0b2;
                            font-size:9px;
                        ">
                            {threat:.0f}
                        </span>

                    </div>

                    <div class="contact-meta">
                        {a["category"]}
                        ·
                        {a["size"]}
                        ·
                        {a["distance"] + i * 4:.1f} m
                    </div>

                    <div class="progress">
                        <div style="
                            width:{likelihood * 100:.0f}%;
                        "></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="good">

                <b style="
                    font-size:9px;
                    color:#50e59b;
                ">
                    ● ARRAY STATUS
                </b>

                <div style="
                    color:#73879b;
                    font-size:8px;
                    margin-top:3px;
                ">
                    6 feature families available ·
                    acoustic pipeline stable
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # RADAR
    # --------------------------------------------------------

    with center:

        st.markdown(
            '<div class="section-title">Acoustic field</div>',
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            radar_fig(a),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            f"""
            <div class="system-strip">

                <div class="system-item">
                    <b>{a["direction"]:03d}°</b>
                    <span>BEARING</span>
                </div>

                <div class="system-item">
                    <b>{a["distance"]:.1f} m</b>
                    <span>RANGE</span>
                </div>

                <div class="system-item">
                    <b>{a["speed"]:.1f} m/s</b>
                    <span>VELOCITY</span>
                </div>

                <div class="system-item">
                    <b>{a["movement"]}</b>
                    <span>MOVEMENT</span>
                </div>

                <div class="system-item">
                    <b>{a["chunk_count"]}</b>
                    <span>INFERENCE WINDOWS</span>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # THREAT
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">Threat assessment</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="eyebrow">
                {a["level"]} PRIORITY
            </div>
            """,
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            threat_gauge(a),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            f"""
            <div class="card-rule"></div>

            <b style="font-size:10px">
                {a["category"]}
            </b>

            <div style="
                color:#73879b;
                font-size:8px;
                margin-top:3px;
            ">
                {a["size"]}
                profile ·
                {a["movement"]}
            </div>

            <div style="
                color:#73879b;
                font-size:8px;
                margin-top:7px;
            ">
                DOA {a["direction"]:03d}°
                ·
                {a["distance"]:.1f} m
                ·
                {a["speed"]:.1f} m/s
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="alert">

                <div class="alert-title">
                    ⚠ REVIEW REQUIRED
                </div>

                <div class="alert-text">
                    Acoustic outputs should be validated against
                    independent sensors before operational action.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Live analytics</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(
        [1.65, 1]
    )

    with c1:

        st.markdown(
            '<div class="card-title">Threat trajectory</div>',
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            trend_fig(
                st.session_state.history
            ),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    with c2:

        history = st.session_state.history

        counts = (
            history["level"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                ]
            )
            .fillna(0)
        )

        fig = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
            )
        )

        fig.update_layout(
            height=270,
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=25,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=False
            ),
            yaxis=dict(
                title="EVENTS",
                gridcolor="rgba(255,255,255,.05)",
            ),
            font=dict(
                color="#8090a3",
                size=8,
            ),
        )

        aeris_plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )


# ============================================================
# TAB 2 — AUDIO LAB
# ============================================================

with tabs[1]:

    st.markdown(
        """
        <div class="ingest">

            <div class="eyebrow">
                AUDIO INGESTION · LONG-FORM ANALYSIS
            </div>

            <div class="ingest-title">
                Acoustic evidence workbench
            </div>

            <div class="ingest-sub">
                Upload recordings up to 500 MB. Long recordings are
                automatically divided into overlapping inference windows
                before feature extraction.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    upload_col, info_col = st.columns(
        [1.45, 1]
    )

    with upload_col:

        uploaded = st.file_uploader(
            "Audio evidence",
            type=[
                "wav",
                "mp3",
                "flac",
                "ogg",
                "m4a",
            ],
            label_visibility="collapsed",
            key="aeris_audio_uploader",
        )

    with info_col:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Ingestion profile
                </div>

                <div class="card-rule"></div>

                <div class="small">
                    MAX FILE
                </div>

                <b>
                    {MAX_UPLOAD_MB} MB
                </b>

                <div style="height:7px"></div>

                <div class="small">
                    CHUNK SIZE
                </div>

                <b>
                    {CHUNK_SECONDS} sec
                </b>

                <div style="height:7px"></div>

                <div class="small">
                    OVERLAP
                </div>

                <b>
                    {CHUNK_OVERLAP_SECONDS} sec
                </b>

                <div style="height:7px"></div>

                <div class="small">
                    TARGET
                </div>

                <b>
                    16 kHz · Mono
                </b>

            </div>
            """,
            unsafe_allow_html=True,
        )

    checkpoint_path = discover_model_checkpoint()
    detected_key = None
    if checkpoint_path:
        try:
            stat = checkpoint_path.stat()
            detected_key = f"{checkpoint_path}::{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            detected_key = str(checkpoint_path)
    _, detected_model_status, detected_threshold = load_repository_model(
        detected_key
    )

    st.markdown(
        f"""
        <div class="system-strip">
            <div class="system-item">
                <b>{MAX_UPLOAD_MB} MB</b>
                <span>UPLOAD CEILING</span>
            </div>
            <div class="system-item">
                <b>{CHUNK_SECONDS}s / {CHUNK_OVERLAP_SECONDS}s</b>
                <span>MODEL WINDOW / HOP</span>
            </div>
            <div class="system-item">
                <b>{"ONLINE" if REPOSITORY_FEATURES_OK else "FALLBACK"}</b>
                <span>REPOSITORY FEATURES</span>
            </div>
            <div class="system-item">
                <b>{"ONLINE" if checkpoint_path else "FALLBACK"}</b>
                <span>MODEL CHECKPOINT</span>
            </div>
            <div class="system-item">
                <b>16 kHz</b>
                <span>MODEL INPUT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source = uploaded

    if source is not None:

        data = source.getvalue()

        size_mb = len(data) / (
            1024 * 1024
        )

        if size_mb > MAX_UPLOAD_MB:

            st.error(
                f"File exceeds the AERIS "
                f"{MAX_UPLOAD_MB} MB application limit."
            )

        else:

            st.session_state.uploaded_name = (
                source.name
            )

            st.session_state.uploaded_size = (
                len(data)
            )

            progress = st.progress(
                0,
                text="Preparing acoustic pipeline…",
            )

            status = st.empty()

            # The number of chunks is not known until decoding.
            # We therefore update progress in a coarse but useful
            # chunk-processing loop.

            def update_progress(n):

                if n <= 1:

                    progress.progress(
                        0.05,
                        text=(
                            "Decoding acoustic evidence…"
                        ),
                    )

                else:

                    # Estimate progress using processed chunk count.
                    # Actual total is determined after decoding.
                    p = min(
                        .92,
                        .08 + n / max(
                            n + 5,
                            1,
                        ) * .84,
                    )

                    progress.progress(
                        p,
                        text=(
                            f"Processing inference "
                            f"window {n}…"
                        ),
                    )

            status.info(
                "Decoding → chunking → feature extraction → inference"
            )

            try:

                result = analyze_audio(
                    data,
                    progress_callback=update_progress,
                )

                st.session_state.analysis = result

                a = result

                progress.progress(
                    1.0,
                    text="Acoustic pipeline complete",
                )

                status.success(
                    f"Processed {a['chunk_count']} "
                    f"inference windows."
                )

            except Exception as exc:

                progress.empty()

                status.error(
                    "Audio processing failed."
                )

                st.exception(exc)

                a = st.session_state.analysis

        if len(data) <= 80 * 1024 * 1024:
            st.audio(
                data,
                format=source.type,
            )
        else:
            st.caption(
                "Playback preview is suppressed for recordings above 80 MB; "
                "the complete file was still processed by the analysis pipeline."
            )

    else:

        st.caption(
            "No recording loaded. AERIS is displaying the deterministic research signal."
        )

    # --------------------------------------------------------
    # AUDIO METRICS
    # --------------------------------------------------------

    m = st.columns(6)

    values = [
        (
            "Duration",
            format_duration(
                a["duration"]
            ),
        ),
        (
            "RMS",
            f"{a['rms']:.4f}",
        ),
        (
            "Peak",
            f"{a['peak']:.3f}",
        ),
        (
            "Centroid",
            f"{a['centroid']/1000:.2f} kHz",
        ),
        (
            "ZCR",
            f"{a['zcr']:.3f}",
        ),
        (
            "Windows",
            str(a["chunk_count"]),
        ),
    ]

    for col, (label, value) in zip(
        m,
        values,
    ):

        with col:

            st.markdown(
                metric_card(
                    label,
                    value,
                    "signal diagnostic",
                ),
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Inference pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-item active">
                <b>DECODE</b>
                <span>audio normalization</span>
            </div>

            <div class="pipeline-item active">
                <b>CHUNK</b>
                <span>4s model windows</span>
            </div>

            <div class="pipeline-item active">
                <b>FEATURE</b>
                <span>MFCC / MEL / spectral</span>
            </div>

            <div class="pipeline-item active">
                <b>MODEL</b>
                <span>repository inference</span>
            </div>

            <div class="pipeline-item active">
                <b>AGGREGATE</b>
                <span>recording-level result</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # VISUALS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Signal intelligence</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.5, 1]
    )

    with left:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Waveform
                </div>

                <div class="card-sub">
                    Downsampled display of the complete recording
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            waveform_fig(a),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Mel spectrogram
                </div>

                <div class="card-sub">
                    Aggregated feature windows across the recording
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            spectrogram_fig(a),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    with right:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Recognition board
                </div>

                <div class="card-sub">
                    Current model / rule-layer outputs
                </div>

                <div class="card-rule"></div>

            """,
            unsafe_allow_html=True,
        )

        rows = [
            (
                "Drone Presence",
                f"{a['drone_likelihood'] * 100:.1f}%",
                "MODEL PROBABILITY" if a.get("model_mean_probability") is not None else "SIGNAL",
            ),
            (
                "Window Decision",
                (
                    "DRONE DETECTED"
                    if a.get("model_window_decision") is True
                    else "NO DRONE WINDOW"
                    if a.get("model_window_decision") is False
                    else "MODEL UNAVAILABLE"
                ),
                (
                    f"THRESHOLD {a.get('decision_threshold') * 100:.1f}%"
                    if a.get("decision_threshold") is not None
                    else "NOT AVAILABLE"
                ),
            ),
            (
                "Drone Category",
                a["category"],
                "MODEL",
            ),
            (
                "Estimated Size",
                a["size"],
                "MODEL",
            ),
            (
                "Flight Movement",
                a["movement"],
                "MODEL",
            ),
            (
                "Direction of Arrival",
                f"{a['direction']:03d}°",
                "EXTENSION",
            ),
            (
                "Estimated Distance",
                f"{a['distance']:.1f} m",
                "EXTENSION",
            ),
            (
                "Estimated Speed",
                f"{a['speed']:.1f} m/s",
                "EXTENSION",
            ),
            (
                "Drone Count",
                str(a["count"]),
                "WINDOW",
            ),
            (
                "Threat Score",
                f"{a['threat']:.0f}/100",
                "RULE",
            ),
            (
                "Threat Level",
                a["level"],
                "RULE",
            ),
        ]

        for name, value, status in rows:

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:8px 0;
                    border-bottom:1px solid rgba(255,255,255,.045);
                ">

                    <span style="
                        color:#73879b;
                        font-size:8px;
                    ">
                        {name}
                    </span>

                    <span>
                        <b style="
                            font-size:9px;
                        ">
                            {value}
                        </b>

                        <span style="
                            color:#6c7e91;
                            font-size:7px;
                            margin-left:5px;
                        ">
                            {status}
                        </span>

                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    if a.get("chunks"):
        window_rows = []
        for chunk in a["chunks"]:
            probability = chunk.get("model_probability")
            window_rows.append(
                {
                    "WINDOW": f"{chunk['index'] + 1:04d}",
                    "START": format_duration(chunk["start"]),
                    "END": format_duration(chunk["end"]),
                    "DRONE PROB.": (
                        f"{probability * 100:.1f}%"
                        if probability is not None
                        else "—"
                    ),
                    "DECISION": (
                        "DRONE"
                        if chunk.get("model_decision") is True
                        else "BACKGROUND"
                        if chunk.get("model_decision") is False
                        else "—"
                    ),
                    "RMS": f"{chunk['rms']:.4f}",
                    "CENTROID": f"{chunk['centroid'] / 1000:.2f} kHz",
                }
            )

        with st.expander(
            f"WINDOW INTELLIGENCE · {len(window_rows)} MODEL WINDOWS",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(window_rows),
                width="stretch",
                height=min(420, 85 + len(window_rows) * 35),
                hide_index=True,
            )

    st.markdown(
        f"""
        <div class="good" style="margin-top:12px">
            <b style="font-size:9px;color:#50e59b">
                ● {a.get("model_status", "ANALYSIS ENGINE")}
            </b>
            <div style="color:#73879b;font-size:8px;margin-top:3px">
                {a["chunk_count"]} model-compatible windows ·
                mean probability:
                {(a.get("model_mean_probability") or 0) * 100:.1f}% ·
                peak:
                {(a.get("model_peak_probability") or 0) * 100:.1f}% ·
                threshold:
                {(a.get("decision_threshold") or 0.5) * 100:.1f}% ·
                positive windows:
                {a.get("model_positive_windows", 0)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HISTORY ACTION
    # --------------------------------------------------------

    st.write("")

    if st.button(
        "＋ ADD ANALYSIS TO DETECTION HISTORY",
        type="primary",
        width="content",
    ):

        row = {
            "timestamp": datetime.now(),
            "source": str(
                a.get(
                    "source",
                    "audio",
                )
            ),
            "category": a["category"],
            "distance_m": round(
                a["distance"],
                1,
            ),
            "speed_mps": round(
                a["speed"],
                1,
            ),
            "threat": round(
                a["threat"],
                1,
            ),
            "confidence": round(
                a["drone_likelihood"],
                3,
            ),
            "level": a["level"],
        }

        st.session_state.history = pd.concat(
            [
                st.session_state.history,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        st.success(
            "Analysis added to local detection history."
        )


# ============================================================
# TAB 3 — AIRSPACE
# ============================================================

with tabs[2]:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                CONTACT GEOMETRY · ACOUSTIC BEARING
            </div>

            <h1>
                Airspace overview
            </h1>

            <p>
                Operator-oriented acoustic spatial visualization.
                Bearing and range are research estimates and should not
                be interpreted as GPS or geospatial truth.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.8, 1]
    )

    with left:

        st.markdown(
            '<div class="section-title">Acoustic field</div>',
            unsafe_allow_html=True,
        )

        aeris_plotly_chart(
            radar_fig(a),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    with right:

        st.markdown(
            '<div class="section-title">Active tracks</div>',
            unsafe_allow_html=True,
        )

        for i in range(
            max(
                1,
                a["count"],
            )
        ):

            d = (
                a["distance"]
                + i * 4
            )

            bearing = (
                a["direction"]
                + i * 37
            ) % 360

            score = max(
                10,
                a["threat"]
                - i * 11,
            )

            level = (
                "CRITICAL"
                if score >= 82
                else "HIGH"
                if score >= 64
                else "MEDIUM"
            )

            st.markdown(
                f"""
                <div class="contact">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                    ">

                        <b class="contact-id">
                            DRONE {i+1:02d}
                        </b>

                        <span style="
                            color:#29f2e2;
                            font-size:8px;
                        ">
                            {level}
                        </span>

                    </div>

                    <div class="contact-meta">
                        Bearing {bearing:03d}°
                        · Range {d:.1f} m
                        · Speed
                        {max(.5, a["speed"] - i * 1.4):.1f} m/s
                    </div>

                    <div class="progress">
                        <div style="
                            width:{max(20, score):.0f}%;
                        "></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="alert">

                <div class="alert-title">
                    SPATIAL LIMITATION
                </div>

                <div class="alert-text">
                    True triangulation requires calibrated
                    microphone-array geometry and/or additional
                    sensors. The visualization is an acoustic
                    bearing/range abstraction.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 4 — HISTORY
# ============================================================

with tabs[3]:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                EVENT ARCHIVE · DETECTION ANALYTICS
            </div>

            <h1>
                Detection history
            </h1>

            <p>
                Review acoustic events, filter threat posture,
                inspect historical confidence and export the
                local research record.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    h = st.session_state.history.copy()

    f1, f2, f3 = st.columns(
        [1, 1, 1.3]
    )

    with f1:

        level = st.selectbox(
            "Threat level",
            [
                "All",
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
        )

    with f2:

        category = st.selectbox(
            "Category",
            [
                "All"
            ]
            + sorted(
                h["category"]
                .unique()
                .tolist()
            ),
        )

    with f3:

        query = st.text_input(
            "Search source",
            placeholder="ARRAY-01",
        )

    if level != "All":
        h = h[
            h.level == level
        ]

    if category != "All":
        h = h[
            h.category == category
        ]

    if query:
        h = h[
            h.source.str.contains(
                query,
                case=False,
                na=False,
            )
        ]

    st.dataframe(
        h.sort_values(
            "timestamp",
            ascending=False,
        ),
        width="stretch",
        height=420,
        hide_index=True,
    )

    st.download_button(
        "EXPORT FILTERED HISTORY",
        h.to_csv(
            index=False
        ).encode(),
        "aeris_detection_history.csv",
        "text/csv",
    )

    st.markdown(
        '<div class="section-title">Archive analytics</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        aeris_plotly_chart(
            trend_fig(
                h
                if len(h)
                else st.session_state.history
            ),
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    with c2:

        counts = (
            h["category"]
            .value_counts()
        )

        fig = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=.62,
            )
        )

        fig.update_layout(
            height=270,
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#8090a3",
                size=8,
            ),
        )

        aeris_plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )


# ============================================================
# TAB 5 — THREAT INTEL
# ============================================================

with tabs[4]:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                THREAT INTELLIGENCE · DECISION SUPPORT
            </div>

            <h1>
                Acoustic threat assessment
            </h1>

            <p>
                Transparent breakdown of the current operator-facing
                threat index and the signal characteristics contributing
                to the score.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(
        [1, 1.7]
    )

    with c1:

        st.markdown(
            f"""
            <div class="card">

                <div class="eyebrow">
                    {a["level"]} PRIORITY
                </div>

                <div class="big-number">
                    {a["threat"]:.0f}
                </div>

                <div style="
                    color:#73879b;
                    font-size:8px;
                ">
                    COMPOSITE INDEX / 100
                </div>

                <div class="card-rule"></div>

                <b style="font-size:10px">
                    {a["category"]}
                </b>

                <div style="
                    color:#73879b;
                    font-size:8px;
                    margin-top:3px;
                ">
                    {a["size"]}
                    ·
                    {a["movement"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        factors = pd.DataFrame(
            {
                "Factor": [
                    "Drone likelihood",
                    "Speed contribution",
                    "Range contribution",
                    "Signal energy",
                ],
                "Contribution": [
                    a["drone_likelihood"]
                    * 62,
                    min(
                        a["speed"] / 28,
                        1,
                    )
                    * 12,
                    (
                        1
                        - min(
                            a["distance"] / 45,
                            1,
                        )
                    )
                    * 18,
                    min(
                        a["rms"] * 180,
                        12,
                    ),
                ],
            }
        )

        fig = go.Figure(
            go.Bar(
                x=factors.Contribution,
                y=factors.Factor,
                orientation="h",
            )
        )

        fig.update_layout(
            height=280,
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                gridcolor="rgba(255,255,255,.05)",
                title="RELATIVE CONTRIBUTION",
            ),
            yaxis=dict(
                showgrid=False
            ),
            font=dict(
                color="#8090a3",
                size=8,
            ),
        )

        aeris_plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    st.markdown(
        """
        <div class="alert">

            <div class="alert-title">
                IMPORTANT MODEL BOUNDARY
            </div>

            <div class="alert-text">
                Extended fields such as category, size, movement,
                direction, distance, speed and threat are currently
                integration-ready outputs. They should only be treated
                as validated model outputs once the corresponding
                repository model heads and calibration are connected.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 6 — MODEL INSIGHTS
# ============================================================

with tabs[5]:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                MODEL OBSERVABILITY · FEATURE INTELLIGENCE
            </div>

            <h1>
                Recognition stack
            </h1>

            <p>
                Inspect the acoustic feature families currently
                represented by AERIS and the integration path toward
                a full multitask recognition system.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    names = [
        "MFCC",
        "Mel Spectrogram",
        "Spectral",
        "Chroma",
        "ZCR",
        "Energy",
    ]

    weights = np.array(
        [
            .22,
            .27,
            .18,
            .12,
            .09,
            .12,
        ]
    )

    if LIBROSA_OK:

        weights *= (
            1
            + np.array(
                [
                    np.std(
                        a["mfcc"]
                    ) / 25,
                    np.std(
                        a["mel"]
                    ) / 20,
                    .2,
                    .1,
                    .1,
                    .15,
                ]
            )
        )

        weights /= weights.sum()

    c1, c2 = st.columns(
        [1.2, 1]
    )

    with c1:

        fig = go.Figure(
            go.Bar(
                x=names,
                y=weights,
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                gridcolor="rgba(255,255,255,.05)",
                title="RELATIVE SIGNAL VARIATION",
            ),
            xaxis=dict(
                showgrid=False
            ),
            font=dict(
                color="#8090a3",
                size=8,
            ),
        )

        aeris_plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

    with c2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Capability matrix
                </div>

                <div class="card-rule"></div>

            """,
            unsafe_allow_html=True,
        )

        capability = [
            (
                "Audio ingestion",
                "READY",
                "500 MB / WAV / MP3 / FLAC / OGG / M4A",
            ),
            (
                "Long recording chunking",
                "READY",
                "4 second overlapping model-compatible inference windows",
            ),
            (
                "Feature extraction",
                "READY",
                "MFCC / Mel / spectral diagnostics",
            ),
            (
                "Drone presence",
                "READY",
                "Current recognition layer",
            ),
            (
                "Drone category",
                "EXTEND",
                "Dedicated class head / checkpoint",
            ),
            (
                "Size + movement",
                "EXTEND",
                "Multitask heads required",
            ),
            (
                "DOA + distance + speed",
                "EXTEND",
                "Calibration + regression heads",
            ),
            (
                "Drone count",
                "EXTEND",
                "Temporal separation / tracking",
            ),
            (
                "Threat score",
                "RULE",
                "Current transparent scoring layer",
            ),
        ]

        for name, status, detail in capability:

            st.markdown(
                f"""
                <div style="
                    padding:7px 0;
                    border-bottom:
                        1px solid rgba(255,255,255,.045);
                ">

                    <b style="
                        font-size:9px;
                    ">
                        {name}
                    </b>

                    <span style="
                        color:#29f2e2;
                        font-size:7px;
                        margin-left:6px;
                    ">
                        {status}
                    </span>

                    <div style="
                        color:#73879b;
                        font-size:7px;
                        margin-top:2px;
                    ">
                        {detail}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Multitask architecture</div>',
        unsafe_allow_html=True,
    )

    st.code(
        """
Microphone Array / Recording
            │
            ▼
      Audio Decode
            │
            ▼
    4s Overlapping Chunks
            │
            ▼
 ┌─────────────────────────┐
 │ MFCC / MEL / SPECTRAL   │
 │ ZCR / ENERGY / CHROMA   │
 └─────────────────────────┘
            │
            ▼
      Feature Encoder
            │
            ▼
        Attention
            │
            ▼
          Fusion
            │
      ┌─────┼──────────┐
      ▼     ▼          ▼
   Presence Category   Size
      │     │          │
      └─────┼──────────┘
            │
     Movement / DOA
            │
       Distance
            │
         Speed
            │
         Count
            │
      Threat Engine
            │
            ▼
   Recording-level result
        """,
        language="text",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:25px;
        padding-top:12px;
        border-top:1px solid rgba(255,255,255,.07);
        color:#506273;
        font-size:7px;
        letter-spacing:.08em;
    ">
        AERIS v3.1 · ACOUSTIC DRONE RECOGNITION & THREAT ASSESSMENT
        · RESEARCH / DEMONSTRATION CONSOLE
    </div>
    """,
    unsafe_allow_html=True,
)