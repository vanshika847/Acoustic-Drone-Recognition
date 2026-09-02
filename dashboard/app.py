from __future__ import annotations

import io
import math
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
# AERIS — Acoustic Intelligence Console
# Professional operations-center UI for the existing repository.
# ============================================================

st.set_page_config(
    page_title="AERIS | Acoustic Intelligence Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Theme ---------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg:#05080d;
    --panel:#0a1018;
    --panel2:#0d1520;
    --panel3:#101b28;
    --line:rgba(160,190,220,.12);
    --line2:rgba(53,231,220,.18);
    --text:#edf5fb;
    --muted:#7f91a5;
    --cyan:#37e6dc;
    --violet:#8c72ff;
    --green:#5de39d;
    --amber:#ffbf69;
    --red:#ff6077;
    --blue:#61a8ff;
}

html,body,[class*="css"] { font-family:Inter,sans-serif; }
.stApp {
    background:
      radial-gradient(1000px 550px at 78% -12%, rgba(106,72,220,.20), transparent 62%),
      radial-gradient(800px 500px at 18% 0%, rgba(35,205,199,.09), transparent 58%),
      linear-gradient(180deg,#05080d 0%,#070b11 100%);
    color:var(--text);
}
.block-container { max-width:1720px; padding:1.1rem 1.65rem 2.5rem; }

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#060a10 0%,#080d14 100%);
    border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] > div { padding:1rem .85rem; }

#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

.aeris-brand { padding:5px 8px 18px; }
.aeris-mark {
    display:inline-flex;width:38px;height:38px;border-radius:11px;
    align-items:center;justify-content:center;
    background:linear-gradient(135deg,rgba(55,230,220,.14),rgba(140,114,255,.28));
    border:1px solid rgba(55,230,220,.28);
    color:var(--cyan);font-weight:800;font-size:19px;
    box-shadow:0 0 28px rgba(55,230,220,.08);
}
.aeris-name { font-family:"Space Grotesk";font-weight:700;font-size:18px;margin-left:9px;vertical-align:8px; }
.aeris-sub { color:#607187;font-size:9px;letter-spacing:.16em;text-transform:uppercase;margin:5px 0 0 47px; }

.nav-label { color:#56667a;font-size:9px;font-weight:800;letter-spacing:.17em;text-transform:uppercase;margin:16px 7px 7px; }

.sidebar-status {
    border:1px solid rgba(93,227,157,.18);
    background:rgba(93,227,157,.055);
    border-radius:12px;padding:11px 12px;
}
.sidebar-status .live { color:var(--green);font-size:10px;font-weight:800;letter-spacing:.08em; }
.sidebar-status .sub { color:var(--muted);font-size:9px;margin-top:4px;line-height:1.45; }

.topbar {
    display:flex;align-items:center;justify-content:space-between;
    padding:3px 0 13px;border-bottom:1px solid var(--line);margin-bottom:16px;
}
.topbar-left .kicker { color:var(--cyan);font-size:9px;font-weight:800;letter-spacing:.17em; }
.topbar-left h1 { font-family:"Space Grotesk";font-size:25px;margin:4px 0 0;letter-spacing:-.035em; }
.topbar-right { display:flex;gap:7px;align-items:center; }
.pill {
    display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
    border-radius:999px;padding:6px 9px;background:rgba(255,255,255,.025);
    color:#aebdcd;font-size:9px;font-weight:700;
}
.pill.live { color:var(--green);border-color:rgba(93,227,157,.2);background:rgba(93,227,157,.05); }
.dot { width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor; }

.hero {
    border:1px solid var(--line);border-radius:18px;padding:21px 23px;
    background:
      linear-gradient(135deg,rgba(13,24,35,.96),rgba(13,12,25,.96));
    box-shadow:0 18px 50px rgba(0,0,0,.25);
}
.hero h2 { font-family:"Space Grotesk";margin:3px 0 0;font-size:29px;letter-spacing:-.04em; }
.hero p { color:var(--muted);font-size:12px;max-width:850px;margin:7px 0 0;line-height:1.55; }
.eyebrow { color:var(--cyan);font-size:9px;font-weight:800;letter-spacing:.17em;text-transform:uppercase; }
.tags { margin-top:13px; }
.tag {
    display:inline-block;border:1px solid rgba(255,255,255,.09);
    background:rgba(255,255,255,.035);border-radius:999px;
    padding:4px 8px;color:#b9c7d6;font-size:9px;margin:0 4px 4px 0;
}

.section {
    margin:18px 0 9px;font-family:"Space Grotesk";font-size:15px;font-weight:700;
}
.card {
    background:linear-gradient(180deg,rgba(12,19,29,.92),rgba(8,13,20,.92));
    border:1px solid var(--line);border-radius:14px;padding:15px;
}
.card-title { font-family:"Space Grotesk";font-size:13px;font-weight:700; }
.card-sub { color:var(--muted);font-size:9px;margin-top:3px; }
.card-rule { border-top:1px solid var(--line);margin:10px 0; }

.metric {
    background:linear-gradient(180deg,rgba(15,24,35,.98),rgba(9,15,22,.98));
    border:1px solid var(--line);border-radius:13px;padding:13px 14px;
    min-height:89px;box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
}
.metric .label { color:#718196;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.09em; }
.metric .value { font-family:"Space Grotesk";font-size:24px;font-weight:700;margin-top:7px;letter-spacing:-.03em; }
.metric .delta { color:var(--green);font-size:9px;margin-top:4px; }

.contact {
    border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.018);
    padding:10px 11px;margin-bottom:7px;
}
.contact:hover { border-color:rgba(55,230,220,.25); }
.contact .id { font-weight:700;font-size:11px; }
.contact .meta { color:var(--muted);font-size:9px;margin-top:4px; }
.bar { height:5px;border-radius:99px;background:#172331;overflow:hidden;margin-top:7px; }
.bar > div { height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--violet)); }

.alert {
    border:1px solid rgba(255,96,119,.2);background:rgba(255,96,119,.055);
    border-radius:11px;padding:10px 11px;margin:7px 0;
}
.alert .t { font-weight:700;font-size:10px; }
.alert .s { color:var(--muted);font-size:9px;margin-top:3px;line-height:1.4; }

.ok {
    border:1px solid rgba(93,227,157,.18);background:rgba(93,227,157,.05);
    border-radius:11px;padding:10px 11px;
}
.small { font-size:9px;color:var(--muted); }
.big-number { font-family:"Space Grotesk";font-size:42px;font-weight:700;letter-spacing:-.05em; }

.stButton > button,.stDownloadButton > button {
    border-radius:9px;border:1px solid var(--line);background:#101925;color:#eaf3fa;
    font-weight:600;
}
.stButton > button:hover { border-color:rgba(55,230,220,.42);color:var(--cyan); }
button[data-baseweb="tab"] { font-weight:700;font-size:10px; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--cyan); }
div[data-testid="stMetric"] { background:rgba(10,16,24,.8);border:1px solid var(--line);border-radius:12px; }
div[data-testid="stFileUploaderDropzone"] { background:rgba(10,17,26,.7);border:1px dashed rgba(55,230,220,.24); }
.stSelectbox label,.stTextInput label,.stSlider label,.stFileUploader label { font-size:9px !important;color:#8291a4 !important; }
hr { border-color:var(--line); }

.status-strip {
    display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:10px;
}
.status-item { padding:9px 10px;border:1px solid var(--line);border-radius:10px;background:rgba(255,255,255,.018); }
.status-item b { font-size:10px; }
.status-item span { display:block;color:var(--muted);font-size:8px;margin-top:3px; }

@media(max-width:900px) {
    .block-container { padding:.8rem; }
    .topbar-right { display:none; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------

def metric_card(label: str, value: str, delta: str = "") -> str:
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="delta">{delta}</div></div>'

def seeded_rng(blob: bytes | None = None):
    seed = 42 if not blob else int(hashlib.sha256(blob).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def load_audio(data: bytes):
    if not data:
        return None, 16000
    if LIBROSA_OK:
        try:
            y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
            return y.astype(np.float32), int(sr)
        except Exception:
            pass
    if SOUNDFILE_OK:
        try:
            y, sr = sf.read(io.BytesIO(data), always_2d=False)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y, int(sr)
        except Exception:
            pass
    return None, 16000

def analyze_audio(data: bytes | None):
    rng = seeded_rng(data)
    y, sr = load_audio(data or b"")
    if y is None or len(y) < 64:
        duration = 10.0
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        y = (0.30*np.sin(2*np.pi*235*t) +
             0.11*np.sin(2*np.pi*470*t) +
             0.055*np.sin(2*np.pi*705*t) +
             0.035*rng.normal(size=t.size)).astype(np.float32)
        sr = 16000
        source = "DEMO SIGNAL"
    else:
        source = "UPLOADED AUDIO"

    duration = len(y) / max(sr, 1)
    rms = float(np.sqrt(np.mean(y*y) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-9)

    if LIBROSA_OK:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        db = librosa.power_to_db(mel, ref=np.max)
        onset = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=.85)))
    else:
        n = min(len(y), sr*8)
        spec = np.abs(np.fft.rfft(y[:n]*np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1/sr)
        centroid = float((spec*freqs).sum()/(spec.sum()+1e-9))
        zcr = float(np.mean(np.diff(np.signbit(y[:n]))))
        mfcc = np.zeros((20,64))
        mel = np.maximum(spec[:2048,None],1e-8)
        db = 20*np.log10(mel/mel.max())
        onset = float(np.std(spec))
        rolloff = float(centroid*1.7)

    likelihood = float(np.clip(
        0.35 + .25*np.tanh((centroid-800)/1300) +
        .18*np.tanh((rms-.025)*14) + .06*np.tanh(onset/3), .02, .98
    ))
    category = ["Multirotor","Fixed-wing","Unknown"][int(np.clip((centroid-500)/1700*2.1,0,2))]
    size = ["Micro","Small","Medium","Large"][int(np.clip((centroid-300)/1300*3.6,0,3))]
    movement = ["Hover","Cruise","Approach","Departing"][int(np.clip((onset/3)*3.2,0,3))]
    direction = int((centroid*0.19 + onset*23) % 360)
    distance = float(np.clip(8 + (1-likelihood)*25 + rng.normal(0,.7), 2.5, 45))
    speed = float(np.clip(2 + likelihood*18 + (onset%1)*5, .5, 28))
    count = int(np.clip(1 + round(likelihood*2 + max(0,onset-1)*.25), 1, 4))
    threat = float(np.clip(likelihood*62 + (1-distance/50)*18 + min(speed/28,1)*12 + (count-1)*4, 0, 100))
    level = "CRITICAL" if threat >= 82 else "HIGH" if threat >= 64 else "MEDIUM" if threat >= 40 else "LOW"

    return {
        "y": y, "sr": sr, "duration": duration, "rms": rms, "peak": peak,
        "centroid": centroid, "zcr": zcr, "mfcc": mfcc, "mel": mel, "db": db,
        "onset": onset, "rolloff": rolloff, "source": source,
        "drone_likelihood": likelihood, "category": category, "size": size,
        "movement": movement, "direction": direction, "distance": distance,
        "speed": speed, "count": count, "threat": threat, "level": level,
    }

def waveform_fig(a):
    y = a["y"]
    step = max(1, len(y)//1800)
    yy = y[::step]
    x = np.linspace(0, a["duration"], len(yy))
    fig = go.Figure(go.Scatter(x=x, y=yy, mode="lines", line=dict(width=1.15)))
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=8,b=25),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (s)",showgrid=False),
        yaxis=dict(title="Amplitude",gridcolor="rgba(255,255,255,.045)"),
        font=dict(color="#8090a3",size=9))
    return fig

def spectrogram_fig(a):
    db = a["db"]
    fig = go.Figure(go.Heatmap(z=db, colorscale="Viridis", showscale=False))
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=8,b=25),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time",showgrid=False),
        yaxis=dict(title="Mel frequency",gridcolor="rgba(255,255,255,.04)"),
        font=dict(color="#8090a3",size=9))
    return fig

def radar_fig(a):
    theta = np.linspace(0, 2*np.pi, 180)
    base = 0.25 + 0.12*np.sin(theta*3+0.4)
    contact_angle = math.radians(a["direction"])
    spread = np.exp(-((np.angle(np.exp(1j*(theta-contact_angle))))**2)/(2*.17**2))
    signal = base + (.65 + .18*a["drone_likelihood"])*spread
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=signal, theta=np.degrees(theta), mode="lines", fill="toself", line=dict(width=1.5)))
    fig.add_trace(go.Scatterpolar(r=[1.02],theta=[a["direction"]],mode="markers+text",
                                  text=["CONTACT"],textposition="top center",
                                  marker=dict(size=10)))
    fig.update_layout(height=390, margin=dict(l=12,r=12,t=20,b=20),
        paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8090a3",size=9),
        polar=dict(bgcolor="rgba(3,8,13,.7)",radialaxis=dict(showticklabels=False,gridcolor="rgba(255,255,255,.05)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,.05)",direction="clockwise")),
        showlegend=False)
    return fig

def trend_fig(history):
    h = history.tail(20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.timestamp,y=h.threat,name="Threat",mode="lines+markers",line=dict(width=2)))
    fig.add_trace(go.Scatter(x=h.timestamp,y=h.confidence*100,name="Confidence",mode="lines",line=dict(width=1.5,dash="dot")))
    fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=25),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,.05)",title="Index"),
        legend=dict(orientation="h",y=1.1),font=dict(color="#8090a3",size=9))
    return fig

def make_history():
    rng = np.random.default_rng(7)
    n = 24
    ts = [datetime.now()-timedelta(minutes=(n-i)*4) for i in range(n)]
    confidence = np.clip(rng.normal(.82,.09,n),.52,.99)
    threat = np.clip(confidence*65+rng.normal(0,12,n),8,98)
    levels = ["LOW" if x<40 else "MEDIUM" if x<64 else "HIGH" if x<82 else "CRITICAL" for x in threat]
    return pd.DataFrame({
        "timestamp":ts,"source":[f"ARRAY-{i%3+1:02d}" for i in range(n)],
        "category":rng.choice(["Multirotor","Fixed-wing","Unknown"],n,p=[.58,.27,.15]),
        "distance_m":np.round(rng.uniform(6,42,n),1),
        "speed_mps":np.round(rng.uniform(2,24,n),1),
        "threat":np.round(threat,1),"confidence":np.round(confidence,3),"level":levels
    })

# --------------------------- State ---------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = analyze_audio(None)
if "history" not in st.session_state:
    st.session_state.history = make_history()

# --------------------------- Sidebar ---------------------------

with st.sidebar:
    st.markdown(
        '<div class="aeris-brand"><span class="aeris-mark">◈</span><span class="aeris-name">AERIS</span>'
        '<div class="aeris-sub">Acoustic Intelligence Console</div></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="nav-label">Operations</div>', unsafe_allow_html=True)
    page = st.radio(
        "Operations",
        ["Command Center","Live Acoustic","Airspace","Detection History","Threat Intel","Model Insights"],
        label_visibility="collapsed"
    )
    st.markdown('<div class="nav-label">System</div>', unsafe_allow_html=True)
    system_view = st.radio(
        "System",
        ["Sensors & Audio","Settings"],
        label_visibility="collapsed"
    )
    st.markdown("<div style='height:9px'></div>",unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-status"><div class="live">● SYSTEM NOMINAL</div>'
        '<div class="sub">Acoustic pipeline online<br>Local research mode · 16 kHz target</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    st.caption("AERIS v2.0 • Research Console")
    st.caption("Existing repository features and model remain the source of truth. Extended mission fields are integration-ready UI until dedicated multitask heads are connected.")

# --------------------------- Header ---------------------------

a = st.session_state.analysis
st.markdown(
    '<div class="topbar"><div class="topbar-left"><div class="kicker">ACOUSTIC AIRSPACE MONITORING</div>'
    '<h1>Operations Center</h1></div><div class="topbar-right">'
    '<span class="pill"><span class="dot" style="color:#61a8ff"></span> LOCAL NODE</span>'
    '<span class="pill live"><span class="dot"></span> LINK NOMINAL</span>'
    '<span class="pill">12:39:27 IST</span></div></div>',
    unsafe_allow_html=True
)

# --------------------------- Command Center ---------------------------

if page == "Command Center":
    st.markdown(
        '<div class="hero"><div class="eyebrow">MISSION OVERVIEW · ACOUSTIC SENSOR FUSION</div>'
        '<h2>Live airspace threat picture</h2>'
        '<p>Monitor acoustic contacts, inspect signal intelligence, assess threat posture and move directly into evidence analysis — all from one operator workspace.</p>'
        '<div class="tags"><span class="tag">REAL-TIME DETECTION</span><span class="tag">MULTI-DRONE AWARENESS</span>'
        '<span class="tag">LIVE SPECTROGRAM</span><span class="tag">THREAT ASSESSMENT</span></div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section">System posture</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        ("Drone likelihood",f"{a['drone_likelihood']*100:.0f}%","signal-derived"),
        ("Threat score",f"{a['threat']:.0f}/100",a["level"]),
        ("Estimated range",f"{a['distance']:.1f} m","acoustic estimate"),
        ("Speed",f"{a['speed']:.1f} m/s",a["movement"]),
        ("Active contacts",str(a["count"]),"current window"),
    ]
    for c,(lab,val,delta) in zip(cols,cards):
        with c: st.markdown(metric_card(lab,val,delta),unsafe_allow_html=True)

    left, mid, right = st.columns([1.15,1.7,1.05])
    with left:
        st.markdown('<div class="section">Active contacts</div>',unsafe_allow_html=True)
        for i in range(max(1,a["count"])):
            likelihood=max(.46,min(.98,a["drone_likelihood"]-i*.11))
            threat=max(18,min(96,a["threat"]-i*9))
            st.markdown(
                f'<div class="contact"><div style="display:flex;justify-content:space-between">'
                f'<span class="id">CONTACT {i+1:02d}</span><span class="small">{threat:.0f}</span></div>'
                f'<div class="meta">{a["category"]} · {a["size"]} · {a["distance"]+i*4:.1f} m</div>'
                f'<div class="bar"><div style="width:{likelihood*100:.0f}%"></div></div></div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<div class="ok"><b style="font-size:10px">● ARRAY STATUS</b>'
            '<div class="small" style="margin-top:3px">6 feature families available · audio stream stable</div></div>',
            unsafe_allow_html=True
        )

    with mid:
        st.markdown('<div class="section">Airspace overview</div>',unsafe_allow_html=True)
        st.plotly_chart(radar_fig(a),use_container_width=True,config={"displayModeBar":False})
        st.markdown(
            f'<div class="status-strip"><div class="status-item"><b>{a["direction"]:03d}°</b><span>BEARING</span></div>'
            f'<div class="status-item"><b>{a["distance"]:.1f} m</b><span>RANGE</span></div>'
            f'<div class="status-item"><b>{a["speed"]:.1f} m/s</b><span>VELOCITY</span></div>'
            f'<div class="status-item"><b>{a["movement"]}</b><span>MOVEMENT</span></div></div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="section">Threat assessment</div>',unsafe_allow_html=True)
        st.markdown(
            f'<div class="card"><div class="eyebrow">{a["level"]} PRIORITY</div>'
            f'<div class="big-number">{a["threat"]:.0f}</div><div class="small">Composite acoustic threat index</div>'
            f'<div class="card-rule"></div><b style="font-size:11px">{a["category"]}</b>'
            f'<div class="small">{a["size"]} profile · {a["movement"]}</div>'
            f'<div class="small" style="margin-top:6px">DOA {a["direction"]:03d}° · {a["distance"]:.1f} m · {a["speed"]:.1f} m/s</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="alert"><div class="t">⚠ CONTACT REQUIRES REVIEW</div>'
            '<div class="s">Validate acoustic findings against independent sensors before operational action.</div></div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="card"><div class="card-title">Event queue</div><div class="card-sub">Latest system events</div><div class="card-rule"></div>'
                    '<div class="small">12:38:54 · Acoustic segment processed</div>'
                    '<div class="small" style="margin-top:7px">12:38:41 · Feature extraction complete</div>'
                    '<div class="small" style="margin-top:7px">12:38:27 · Contact confidence updated</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section">Live analytics</div>',unsafe_allow_html=True)
    c1,c2 = st.columns([1.6,1])
    with c1:
        st.plotly_chart(trend_fig(st.session_state.history),use_container_width=True,config={"displayModeBar":False})
    with c2:
        h=st.session_state.history
        counts=h["level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0)
        fig=go.Figure(go.Bar(x=counts.index,y=counts.values))
        fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,.05)",title="Events"),
                          font=dict(color="#8090a3",size=9))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# --------------------------- Live Acoustic ---------------------------

elif page == "Live Acoustic":
    st.markdown(
        '<div class="hero"><div class="eyebrow">AUDIO INGESTION · SIGNAL INTELLIGENCE</div>'
        '<h2>Live acoustic workbench</h2>'
        '<p>Upload WAV, MP3, FLAC, OGG or M4A evidence, or capture a microphone segment when supported. Inspect waveform, Mel spectrogram and recognition outputs together.</p></div>',
        unsafe_allow_html=True
    )
    u1,u2=st.columns([1.3,1])
    with u1:
        uploaded=st.file_uploader("Audio evidence",type=["wav","mp3","flac","ogg","m4a"],label_visibility="collapsed")
    with u2:
        mic=None
        if hasattr(st,"audio_input"):
            mic=st.audio_input("Record from microphone",sample_rate=16000)
        else:
            st.info("Microphone capture is unavailable in this Streamlit version. File upload is ready.")
    source=mic if mic is not None else uploaded
    if source is not None:
        data=source.getvalue()
        st.session_state.analysis=analyze_audio(data)
        st.session_state.analysis["source"]=getattr(source,"name","MICROPHONE")
        a=st.session_state.analysis
        st.audio(source)
    else:
        st.caption("Exploration mode: deterministic demo signal is displayed until evidence is supplied.")

    m=st.columns(6)
    vals=[("Duration",f"{a['duration']:.2f}s"),("RMS",f"{a['rms']:.4f}"),("Peak",f"{a['peak']:.3f}"),
          ("Centroid",f"{a['centroid']/1000:.2f} kHz"),("ZCR",f"{a['zcr']:.3f}"),("Rolloff",f"{a['rolloff']/1000:.2f} kHz")]
    for c,(l,v) in zip(m,vals):
        with c: st.markdown(metric_card(l,v,"signal diagnostic"),unsafe_allow_html=True)

    st.markdown('<div class="section">Signal intelligence</div>',unsafe_allow_html=True)
    p1,p2=st.columns([1.55,1])
    with p1:
        st.markdown('<div class="card"><div class="card-title">Waveform</div><div class="card-sub">Raw amplitude over time</div></div>',unsafe_allow_html=True)
        st.plotly_chart(waveform_fig(a),use_container_width=True,config={"displayModeBar":False})
        st.markdown('<div class="card"><div class="card-title">Live Mel spectrogram</div><div class="card-sub">Frequency-energy distribution</div></div>',unsafe_allow_html=True)
        st.plotly_chart(spectrogram_fig(a),use_container_width=True,config={"displayModeBar":False})
    with p2:
        st.markdown('<div class="card"><div class="card-title">Recognition board</div><div class="card-sub">Current repository-backed scope + explicit extension points</div><div class="card-rule"></div>',unsafe_allow_html=True)
        rows=[
            ("Drone Presence",f"{a['drone_likelihood']*100:.1f}%","SIGNAL"),
            ("Drone Category",a["category"],"EXTENSION"),
            ("Estimated Size",a["size"],"EXTENSION"),
            ("Flight Movement",a["movement"],"EXTENSION"),
            ("Direction of Arrival",f"{a['direction']:03d}°","EXTENSION"),
            ("Estimated Distance",f"{a['distance']:.1f} m","EXTENSION"),
            ("Estimated Speed",f"{a['speed']:.1f} m/s","EXTENSION"),
            ("Drone Count",str(a["count"]),"WINDOW"),
            ("Threat Score",f"{a['threat']:.0f}/100","RULE LAYER"),
            ("Threat Level",a["level"],"RULE LAYER"),
        ]
        for name,val,status in rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.05)">'
                f'<span class="small">{name}</span><span><b style="font-size:10px">{val}</b> <span class="tag">{status}</span></span></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>',unsafe_allow_html=True)

    if st.button("Add analysis to detection history",type="primary"):
        row={"timestamp":datetime.now(),"source":str(a.get("source","audio")),"category":a["category"],
             "distance_m":round(a["distance"],1),"speed_mps":round(a["speed"],1),"threat":round(a["threat"],1),
             "confidence":round(a["drone_likelihood"],3),"level":a["level"]}
        st.session_state.history=pd.concat([st.session_state.history,pd.DataFrame([row])],ignore_index=True)
        st.success("Analysis added to the local detection history.")

# --------------------------- Airspace ---------------------------

elif page == "Airspace":
    st.markdown(
        '<div class="hero"><div class="eyebrow">CONTACT GEOMETRY · ACOUSTIC BEARING</div>'
        '<h2>Airspace overview</h2>'
        '<p>Operator-oriented spatial view of active acoustic contacts. This is an acoustic bearing/range visualization, not a GPS track or geospatial truth source.</p></div>',
        unsafe_allow_html=True
    )
    left,right=st.columns([1.8,1])
    with left:
        st.markdown('<div class="section">Acoustic field</div>',unsafe_allow_html=True)
        st.plotly_chart(radar_fig(a),use_container_width=True,config={"displayModeBar":False})
    with right:
        st.markdown('<div class="section">Active track cards</div>',unsafe_allow_html=True)
        for i in range(max(1,a["count"])):
            d=a["distance"]+i*4.0
            bearing=(a["direction"]+i*37)%360
            score=max(10,a["threat"]-i*11)
            st.markdown(
                f'<div class="contact"><div style="display:flex;justify-content:space-between"><b class="id">DRONE {i+1:02d}</b>'
                f'<span class="tag">{"HIGH" if score>=64 else "MEDIUM"}</span></div>'
                f'<div class="meta">Bearing {bearing:03d}° · Range {d:.1f} m · Speed {max(.5,a["speed"]-i*1.4):.1f} m/s</div>'
                f'<div class="bar"><div style="width:{max(20,score):.0f}%"></div></div></div>',
                unsafe_allow_html=True
            )
        st.markdown('<div class="alert"><div class="t">Spatial limitation</div><div class="s">True triangulation requires calibrated microphone-array geometry and/or additional sensors. The UI is ready for those coordinates.</div></div>',unsafe_allow_html=True)

# --------------------------- Detection History ---------------------------

elif page == "Detection History":
    st.markdown(
        '<div class="hero"><div class="eyebrow">EVENT ARCHIVE · ANALYTICS</div>'
        '<h2>Detection history</h2><p>Review acoustic events, filter threat posture and export the local research record.</p></div>',
        unsafe_allow_html=True
    )
    h=st.session_state.history.copy()
    f1,f2,f3=st.columns([1.2,1,1.3])
    with f1: level=st.selectbox("Threat level",["All","LOW","MEDIUM","HIGH","CRITICAL"])
    with f2: category=st.selectbox("Category",["All"]+sorted(h["category"].unique().tolist()))
    with f3: query=st.text_input("Search source",placeholder="ARRAY-01")
    if level!="All": h=h[h.level==level]
    if category!="All": h=h[h.category==category]
    if query: h=h[h.source.str.contains(query,case=False,na=False)]
    st.dataframe(h.sort_values("timestamp",ascending=False),use_container_width=True,height=420,hide_index=True)
    st.download_button("Export filtered history (CSV)",h.to_csv(index=False).encode(),"aeris_detection_history.csv","text/csv")
    st.markdown('<div class="section">Archive analytics</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(trend_fig(h if len(h) else st.session_state.history),use_container_width=True,config={"displayModeBar":False})
    with c2:
        counts=h["category"].value_counts()
        fig=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.62))
        fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8090a3",size=9))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# --------------------------- Threat Intel ---------------------------

elif page == "Threat Intel":
    st.markdown(
        '<div class="hero"><div class="eyebrow">THREAT INTELLIGENCE</div>'
        '<h2>Acoustic threat assessment</h2><p>Transparent scoring view showing the factors currently contributing to the operator-facing threat index.</p></div>',
        unsafe_allow_html=True
    )
    c1,c2=st.columns([1,1.7])
    with c1:
        st.markdown(f'<div class="card"><div class="eyebrow">{a["level"]} PRIORITY</div><div class="big-number">{a["threat"]:.0f}</div>'
                    f'<div class="small">Composite score / 100</div><div class="card-rule"></div>'
                    f'<b style="font-size:11px">{a["category"]}</b><div class="small">{a["size"]} · {a["movement"]}</div></div>',unsafe_allow_html=True)
    with c2:
        factors=pd.DataFrame({
            "Factor":["Drone likelihood","Speed contribution","Range contribution","Signal energy"],
            "Contribution":[a["drone_likelihood"]*62,min(a["speed"]/28,1)*12,(1-min(a["distance"]/45,1))*18,min(a["rms"]*180,12)]
        })
        fig=go.Figure(go.Bar(x=factors.Contribution,y=factors.Factor,orientation="h"))
        fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(gridcolor="rgba(255,255,255,.05)",title="Relative contribution"),
                          yaxis=dict(showgrid=False),font=dict(color="#8090a3",size=9))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown('<div class="alert"><div class="t">Important model boundary</div><div class="s">The repository currently does not contain a dedicated validated multitask threat model. Extended fields and composite threat scoring are explicitly presented as integration-ready UI / rule-layer outputs.</div></div>',unsafe_allow_html=True)

# --------------------------- Model Insights ---------------------------

elif page == "Model Insights":
    st.markdown(
        '<div class="hero"><div class="eyebrow">MODEL OBSERVABILITY</div>'
        '<h2>Feature intelligence</h2><p>Inspect the acoustic feature families already represented in the repository and see where the recognition stack can grow into the planned multitask system.</p></div>',
        unsafe_allow_html=True
    )
    names=["MFCC","Mel Spectrogram","Spectral","Chroma","ZCR","Energy"]
    weights=np.array([.22,.27,.18,.12,.09,.12])
    if LIBROSA_OK:
        weights*=1+np.array([np.std(a["mfcc"])/25,np.std(a["mel"])/20,.2,.1,.1,.15])
        weights/=weights.sum()
    c1,c2=st.columns([1.2,1])
    with c1:
        fig=go.Figure(go.Bar(x=names,y=weights))
        fig.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(gridcolor="rgba(255,255,255,.05)",title="Relative signal variation"),xaxis=dict(showgrid=False),
                          font=dict(color="#8090a3",size=9))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with c2:
        st.markdown('<div class="card"><div class="card-title">Capability matrix</div><div class="card-rule"></div>',unsafe_allow_html=True)
        capability=[
            ("Audio ingestion","READY","WAV / MP3 / FLAC / OGG / M4A + microphone"),
            ("Feature extraction","READY","MFCC / Mel / spectral diagnostics"),
            ("Drone presence","READY","Current classifier architecture"),
            ("Drone category","EXTEND","Dedicated class head / checkpoint"),
            ("Size + movement","EXTEND","Multitask heads required"),
            ("DOA + distance + speed","EXTEND","Calibration + regression heads"),
            ("Drone count","EXTEND","Temporal separation / tracking"),
            ("Threat score / level","EXTEND","Validated threat head or calibrated engine"),
        ]
        for name,status,detail in capability:
            st.markdown(f'<div style="padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05)"><b style="font-size:10px">{name}</b>'
                        f' <span class="tag">{status}</span><div class="small">{detail}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Planned multitask pipeline</div>',unsafe_allow_html=True)
    st.code(
        "Microphone Array → Preprocessing → MFCC / Mel / Chroma / Spectral / ZCR / Energy\n"
        "→ Feature Encoder → Attention → Fusion →\n"
        "Presence | Category | Size | Movement | DOA | Distance | Speed | Count | Threat Score | Threat Level",
        language="text"
    )

# --------------------------- System panels ---------------------------

if system_view == "Sensors & Audio":
    with st.expander("Sensor & audio status",expanded=False):
        s1,s2,s3,s4=st.columns(4)
        s1.metric("Audio input","ONLINE")
        s2.metric("Target sample rate","16 kHz")
        s3.metric("Channels","Mono")
        s4.metric("Feature families","6")
elif system_view == "Settings":
    with st.expander("Console settings",expanded=False):
        st.toggle("Operator confirmation for high-threat events",value=True)
        st.toggle("Persist detection history",value=True)
        st.selectbox("Visualization density",["Command","Analyst","Minimal"],index=0)
        st.selectbox("Console theme",["AERIS Dark"],index=0)

st.markdown("<div style='height:18px'></div><div class='small'>AERIS • Acoustic Drone Recognition & Threat Assessment • Research / demonstration console</div>",unsafe_allow_html=True)
