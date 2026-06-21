"""
app.py — Aplikasi Streamlit: Prediksi Kehadiran Anggota
PT Cahaya Ladara Nusantara

Jalankan dari root folder skripsi/:
    streamlit run app.py
"""

import os
import io
import json
import math
import base64
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image
from fpdf import FPDF

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Kehadiran Anggota | PT CLN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONSTANTA PATH
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "processed", "dataset_gabungan.csv")
MODEL_PKL = os.path.join(ROOT, "models", "random_forest_model.pkl")
METRICS_JSON = os.path.join(ROOT, "outputs", "metrics_summary.json")
IMG_CM = os.path.join(ROOT, "outputs", "confusion_matrix.png")
IMG_FI = os.path.join(ROOT, "outputs", "feature_importance.png")
IMG_DK   = os.path.join(ROOT, "outputs", "distribusi_kehadiran.png")
LOGO_PATH = os.path.join(ROOT, "logo.png")

# Deteksi environment: lokal vs Streamlit Cloud (/mount/src hanya ada di Cloud)
IS_LOCAL = not os.path.exists('/mount/src')
LOG_JSON = os.path.join(ROOT, "pred_log.json")

FEATURE_COLS = [
    "Jenis_Kelamin", "Usia", "Jarak_km",
    "Status_Pendaftaran", "Event_ID", "hadir_event_sebelumnya",
]

# ============================================================
# CSS Ã¢â‚¬â€ MONOCHROME CLEAN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Ã¢â€â‚¬Ã¢â€â‚¬ Base Ã¢â€â‚¬Ã¢â€â‚¬ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Background Ã¢â€â‚¬Ã¢â€â‚¬ */
.stApp {
    background-color: #eef1f8;
    background-image: radial-gradient(#c8d5e8 1px, transparent 1px);
    background-size: 28px 28px;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Main block padding Ã¢â€â‚¬Ã¢â€â‚¬ */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Sidebar Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stSidebar"] {
    background-color: #0f2040;
    color: #ffffff;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #c8d5e8 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* ── Sidebar toggle button ── */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    width: 34px !important;
    height: 34px !important;
    transition: background 0.2s ease !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255,255,255,0.24) !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] > div:first-child {
    overflow-y: hidden !important;
    padding: 0 !important;
}



/* Ã¢â€â‚¬Ã¢â€â‚¬ Sidebar logo Ã¢â€â‚¬Ã¢â€â‚¬ */
.sidebar-logo {
    padding: 16px 16px 14px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
.sidebar-logo img {
    width: 75%;
    height: auto;
    display: inline-block;
    pointer-events: none;
    user-select: none;
    -webkit-user-drag: none;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Sidebar sections Ã¢â€â‚¬Ã¢â€â‚¬ */
.sidebar-section {
    padding: 13px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #3d6491 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.sidebar-title {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin-bottom: 2px;
    letter-spacing: 0.2px;
}
.sidebar-sub {
    font-size: 12px !important;
    color: #6a94be !important;
}
.sidebar-metric {
    font-size: 34px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    line-height: 1;
    letter-spacing: -1px;
}
.sidebar-info {
    font-size: 13px !important;
    color: #c8d5e8 !important;
    line-height: 1.85;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Tab bar Ã¢â‚¬â€œ pill style Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-baseweb="tab-list"] {
    gap: 6px;
    background: #dde4f0;
    border: none;
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 20px;
}
[data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px;
    color: #6b7fa0;
    font-weight: 600;
    padding: 10px 22px;
    font-size: 16px !important;
    transition: all 0.2s ease;
}
[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.5);
    color: #0f2040;
}
[aria-selected="true"] {
    background: #0f2040 !important;
    color: #ffffff !important;
    border-bottom: none !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(15,32,64,0.3) !important;
    border-radius: 8px !important;
}
[aria-selected="true"] p,
[aria-selected="true"] span,
[aria-selected="true"] div,
[aria-selected="true"] * {
    color: #ffffff !important;
}
[data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-baseweb="tab-border"] {
    display: none !important;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Metric cards Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="metric-container"] {
    background: #ffffff;
    border: none;
    border-radius: 12px;
    border-top: 3px solid #2563a8;
    padding: 20px 22px 18px 22px;
    box-shadow: 0 2px 12px rgba(15,32,64,0.08);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 24px rgba(15,32,64,0.14);
    transform: translateY(-2px);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 13px !important;
    color: #4a6fa5 !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 800 !important;
    color: #0f2040 !important;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Form / widget labels Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label,
.stRadio label,
.stCheckbox label {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #0f2040 !important;
    margin-bottom: 4px !important;
}
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-baseweb="input"] input {
    font-size: 17px !important;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Buttons Ã¢â€â‚¬Ã¢â€â‚¬ */
.stButton > button {
    background: #0f2040;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 17px;
    padding: 12px 28px;
    transition: all 0.2s ease;
    box-shadow: 0 3px 10px rgba(15,32,64,0.22);
    letter-spacing: 0.2px;
}
.stButton > button *,
.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
}
.stButton > button:hover {
    background: #1a3360;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(15,32,64,0.32);
    color: #ffffff !important;
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(15,32,64,0.2);
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Download button Ã¢â€â‚¬Ã¢â€â‚¬ */
.stDownloadButton > button {
    background: #ffffff;
    color: #0f2040;
    border: 2px solid #0f2040;
    border-radius: 8px;
    font-weight: 700;
    font-size: 17px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(15,32,64,0.08);
}
.stDownloadButton > button:hover {
    background: #eef1f8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,32,64,0.15);
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Prediksi result boxes Ã¢â€â‚¬Ã¢â€â‚¬ */
.result-hadir {
    background: linear-gradient(135deg, #0a1828 0%, #0f2040 50%, #163566 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 32px 30px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(15,32,64,0.35), inset 0 1px 0 rgba(255,255,255,0.1);
}
.result-tidak-hadir {
    background: #ffffff;
    color: #0f2040;
    border: 2px solid #0f2040;
    border-radius: 12px;
    padding: 32px 30px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 16px 0;
    box-shadow: 0 4px 16px rgba(15,32,64,0.1);
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Section header Ã¢â€â‚¬Ã¢â€â‚¬ */
.section-header {
    font-size: 26px;
    font-weight: 800;
    color: #0f2040;
    margin-bottom: 16px;
    padding-left: 14px;
    border-left: 4px solid #2563a8;
    line-height: 1.2;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Info box Ã¢â€â‚¬Ã¢â€â‚¬ */
.info-box {
    background: #ffffff;
    border: none;
    border-left: 4px solid #c8d5e8;
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
    margin: 8px 0;
    font-size: 16px;
    color: #1e2d4a;
    line-height: 1.75;
    box-shadow: 0 2px 8px rgba(15,32,64,0.05);
    transition: border-color 0.2s ease;
}
.info-box:hover {
    border-left-color: #2563a8;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Metric table Ã¢â€â‚¬Ã¢â€â‚¬ */
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 16px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(15,32,64,0.08);
}
.metric-table th {
    background: #0f2040;
    color: #ffffff;
    padding: 14px 18px;
    text-align: left;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.3px;
}
.metric-table td {
    padding: 13px 18px;
    border-bottom: 1px solid #eef1f8;
    color: #1e2d4a;
    font-size: 15px;
    line-height: 1.5;
    background: #ffffff;
}
.metric-table tr:nth-child(even) td {
    background: #f5f7fc;
}
.metric-table tr:hover td {
    background: #edf2fc;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ General markdown / paragraph text Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 16px;
    line-height: 1.75;
    color: #1e2d4a;
}
[data-testid="stMarkdownContainer"] b,
[data-testid="stMarkdownContainer"] strong {
    font-weight: 700;
    color: #0f2040;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Hide streamlit branding Ã¢â€â‚¬Ã¢â€â‚¬ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
header button {visibility: visible !important;}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Input widget styling Ã¢â€â‚¬Ã¢â€â‚¬ */
div[data-baseweb="select"] > div {
    border: 2px solid #c8d5e8 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-baseweb="input"] > div {
    border: 2px solid #c8d5e8 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="input"]:focus-within > div {
    border: 2px solid #2563a8 !important;
    box-shadow: 0 0 0 3px rgba(37,99,168,0.15) !important;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Expander Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dde4f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(15,32,64,0.05);
    overflow: hidden;
}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Plotly/chart container Ã¢â€â‚¬Ã¢â€â‚¬ */
[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 2px 10px rgba(15,32,64,0.07);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGIN GATE
# ============================================================
def _check_login(username: str, password: str) -> bool:
    try:
        valid_user = st.secrets["credentials"]["username"]
        valid_pass = st.secrets["credentials"]["password"]
    except Exception:
        valid_user = "admin"
        valid_pass = "supervisor"
    return username.strip() == valid_user and password == valid_pass


if not st.session_state.get("logged_in", False):
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_login, _ = st.columns([1, 1.4, 1])

    with col_login:
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as _lf:
                _b64 = base64.b64encode(_lf.read()).decode()
            st.markdown(
                f'<div style="text-align:center;margin-bottom:20px;">'
                f'<img src="data:image/png;base64,{_b64}" '
                f'style="width:120px;height:auto;" /></div>',
                unsafe_allow_html=True
            )

        st.markdown("""
        <div style="background:#ffffff;border-radius:16px;
                    padding:36px 36px 8px 36px;
                    box-shadow:0 8px 40px rgba(15,32,64,0.14);
                    border-top:4px solid #2563a8;margin-bottom:0;">
            <div style="font-size:21px;font-weight:800;color:#0f2040;
                        text-align:center;margin-bottom:4px;">
                Sistem Prediksi Kehadiran
            </div>
            <div style="font-size:13px;color:#6b7fa0;text-align:center;
                        margin-bottom:24px;font-weight:500;">
                PT Cahaya Ladara Nusantara &nbsp;|&nbsp; Login Admin
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_input(
            "Username", placeholder="Masukkan username", key="login_user"
        )
        pass_input = st.text_input(
            "Password", type="password",
            placeholder="Masukkan password", key="login_pass"
        )
        if st.button("Masuk", use_container_width=True, key="login_btn"):
            if _check_login(user_input, pass_input):
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Username atau password salah.")

    st.stop()


# ============================================================
# LOAD DATA (cached)
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_CSV)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PKL)


@st.cache_data
def load_metrics():
    with open(METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_log_from_disk():
    """Load persisted prediction log from disk. Localhost only."""
    if IS_LOCAL and os.path.exists(LOG_JSON):
        try:
            with open(LOG_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log_to_disk(log):
    """Write prediction log to disk. Localhost only — silently skipped on Cloud."""
    if IS_LOCAL:
        try:
            with open(LOG_JSON, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as _lf:
            _logo_b64 = base64.b64encode(_lf.read()).decode()
        st.markdown(
            '<div class="sidebar-logo">'
            f'<img src="data:image/png;base64,{_logo_b64}" />'
            '</div>',
            unsafe_allow_html=True
        )

    try:
        _acc = f"{load_metrics()['accuracy_pct']:.1f}%"
    except Exception:
        _acc = "N/A"

    st.markdown(f"""
    <div>
        <div class="sidebar-section">
            <div class="sidebar-title">Prediksi Kehadiran</div>
            <div class="sidebar-sub">PT Cahaya Ladara Nusantara</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">Akurasi Model</div>
            <div class="sidebar-metric">{_acc}</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">Pelatihan</div>
            <div class="sidebar-info">Makanan Viral<br>Sushi &amp; Onigiri</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">Jadwal Event</div>
            <div class="sidebar-info">
                Event 1 &nbsp;&nbsp; 09 Nov 2025<br>
                Event 2 &nbsp;&nbsp; 16 Nov 2025
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:16px 12px 8px 12px;'>", unsafe_allow_html=True)
    if st.button("Keluar", use_container_width=True, key="logout_btn"):
        st.session_state["logged_in"] = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# JUDUL
# ============================================================
st.markdown("""
<div style="background:#0f2040;border-radius:12px;padding:28px 36px 24px 36px;margin-bottom:16px;">
  <div style="font-size:28px;font-weight:800;color:#ffffff;margin-bottom:8px;">Sistem Prediksi Kehadiran Anggota</div>
  <div style="font-size:17px;color:#c8d5e8;font-weight:500;">Algoritma Random Forest, Pelatihan Sushi &amp; Onigiri &nbsp;|&nbsp; PT Cahaya Ladara Nusantara</div>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_dash, tab_eval, tab_log = st.tabs(["Prediksi Kehadiran", "Dashboard", "Evaluasi Model", "Log Prediksi"])

# Ã¢â€â‚¬Ã¢â€â‚¬ Session state init Ã¢â€â‚¬Ã¢â€â‚¬
if "pred_log" not in st.session_state:
    # On localhost: restore from disk so log survives app restarts
    st.session_state["pred_log"] = _load_log_from_disk()

# ============================================================
# TAB DASHBOARD
# ============================================================
with tab_dash:
    st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)

    try:
        df = load_dataset()

        # ---- Metric Cards ----
        total = len(df)
        hadir = (df["Status_Kehadiran"] == 1).sum()
        tidak_hadir = (df["Status_Kehadiran"] == 0).sum()
        pct_hadir = hadir / total * 100
        ev1 = df[df["Event_ID"] == 1]
        ev2 = df[df["Event_ID"] == 2]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Peserta", f"{total:,}")
        c2.metric("Hadir", f"{hadir:,}")
        c3.metric("Tidak Hadir", f"{tidak_hadir:,}")
        c4.metric("Tingkat Kehadiran", f"{pct_hadir:.1f}%")
        c5.metric("Jumlah Event", "2")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Chart Row 1: Distribusi Kehadiran + Gender ----
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Distribusi Status Kehadiran per Event**")
            ev_labels = ["Event 1 (09 Nov)", "Event 2 (16 Nov)"]
            hadir_vals = [
                (ev1["Status_Kehadiran"] == 1).sum(),
                (ev2["Status_Kehadiran"] == 1).sum(),
            ]
            tidak_vals = [
                (ev1["Status_Kehadiran"] == 0).sum(),
                (ev2["Status_Kehadiran"] == 0).sum(),
            ]
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(
                name="Hadir", x=ev_labels, y=hadir_vals,
                marker_color="#0f2040", text=hadir_vals,
                textposition="outside"
            ))
            fig_ev.add_trace(go.Bar(
                name="Tidak Hadir", x=ev_labels, y=tidak_vals,
                marker_color="#6b9ed4", text=tidak_vals,
                textposition="outside"
            ))
            fig_ev.update_layout(
                barmode="group", height=320,
                margin=dict(t=20, b=20, l=0, r=0),
                legend=dict(orientation="h", y=-0.15),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(gridcolor="#f0f0f0"),
                font=dict(family="Inter", size=12),
            )
            st.plotly_chart(fig_ev, use_container_width=True)

        with col_r:
            st.markdown("**Distribusi Jenis Kelamin**")
            jk_map = {0: "Perempuan", 1: "Laki-Laki"}
            jk_counts = df["Jenis_Kelamin"].map(jk_map).value_counts()
            fig_jk = go.Figure(go.Pie(
                labels=jk_counts.index.tolist(),
                values=jk_counts.values.tolist(),
                hole=0.5,
                marker=dict(colors=["#0f2040", "#6b9ed4"]),
                textinfo="label+percent",
                textfont=dict(size=13),
            ))
            fig_jk.update_layout(
                height=320,
                margin=dict(t=20, b=20, l=0, r=0),
                showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_jk, use_container_width=True)

        # ---- Chart Row 2: Usia + Jarak ----
        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown("**Distribusi Usia Peserta**")
            fig_usia = go.Figure()
            fig_usia.add_trace(go.Histogram(
                x=df["Usia"], nbinsx=15,
                marker_color="#1a3360", marker_line_color="#ffffff",
                marker_line_width=1,
                name="Usia"
            ))
            fig_usia.add_vline(
                x=df["Usia"].mean(), line_dash="dash",
                line_color="#0f2040", line_width=2,
                annotation_text=f"Rata-rata: {df['Usia'].mean():.1f}",
                annotation_position="top right",
            )
            fig_usia.update_layout(
                height=300, showlegend=False,
                margin=dict(t=20, b=20, l=0, r=0),
                xaxis_title="Usia (tahun)", yaxis_title="Frekuensi",
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(gridcolor="#f0f0f0"),
                font=dict(family="Inter", size=12),
            )
            st.plotly_chart(fig_usia, use_container_width=True)

        with col_r2:
            st.markdown("**Distribusi Jarak ke Lokasi Event (km)**")
            jarak_plot = df["Jarak_km"].clip(upper=25)
            fig_jarak = go.Figure()
            fig_jarak.add_trace(go.Histogram(
                x=jarak_plot, nbinsx=20,
                marker_color="#1a3360", marker_line_color="#ffffff",
                marker_line_width=1,
                name="Jarak"
            ))
            fig_jarak.add_vline(
                x=df["Jarak_km"].median(), line_dash="dash",
                line_color="#0f2040", line_width=2,
                annotation_text=f"Median: {df['Jarak_km'].median():.1f} km",
                annotation_position="top right",
            )
            fig_jarak.update_layout(
                height=300, showlegend=False,
                margin=dict(t=20, b=20, l=0, r=0),
                xaxis_title="Jarak (km, di-cap pada 25 km)",
                yaxis_title="Frekuensi",
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(gridcolor="#f0f0f0"),
                font=dict(family="Inter", size=12),
            )
            st.plotly_chart(fig_jarak, use_container_width=True)

        # ---- Info Statistik ----
        st.markdown('<div class="section-header" style="margin-top:12px;">Statistik Deskriptif</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""<div class="info-box">
            <b>Usia</b><br>
            Rata-rata: {df['Usia'].mean():.1f} tahun<br>
            Std: {df['Usia'].std():.1f} | Min: {int(df['Usia'].min())} | Max: {int(df['Usia'].max())}
            </div>""", unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""<div class="info-box">
            <b>Jarak ke Lokasi</b><br>
            Rata-rata: {df['Jarak_km'].mean():.2f} km<br>
            Median: {df['Jarak_km'].median():.2f} km | Max: {df['Jarak_km'].max():.2f} km
            </div>""", unsafe_allow_html=True)
        with col_s3:
            n_overlap = (df[df["Event_ID"] == 2]["hadir_event_sebelumnya"] == 1).sum()
            st.markdown(f"""<div class="info-box">
            <b>Riwayat Lintas Event</b><br>
            Peserta Event 2 yang hadir di Event 1: <b>{n_overlap}</b><br>
            ({n_overlap / len(ev2) * 100:.1f}% dari peserta Event 2)
            </div>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Jalankan `python scripts/01_preprocess.py` terlebih dahulu.")


# ============================================================
# TAB PREDIKSI KEHADIRAN (landing tab)
# ============================================================
with tab_pred:
    st.markdown('<div class="section-header">Prediksi Kehadiran Peserta Baru</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4a6fa5;font-size:17px;font-weight:600;'>Masukkan data peserta untuk mendapatkan prediksi kehadiran beserta analisis faktor yang memengaruhinya.</p>",
        unsafe_allow_html=True
    )

    try:
        model = load_model()

        # ---- Session state for prediction persistence (fixes CSV download) ----
        if 'pred_data' not in st.session_state:
            st.session_state['pred_data'] = None

        # ---- Layout: Form (left) | Quick Result (right) ----
        col_form, col_result = st.columns([5, 5], gap="large")

        with col_form:

            st.markdown("**Data Peserta**")
            jk = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-Laki"], key="pred_jk")
            usia = st.number_input(
                "Usia (tahun)", min_value=18, max_value=75, value=40, step=1, key="pred_usia"
            )
            jarak = st.number_input(
                "Jarak dari Rumah ke Lokasi Event (km)",
                min_value=0.1, max_value=60.0, value=5.0, step=0.1, format="%.1f", key="pred_jarak"
            )
            event = st.selectbox(
                "Event",
                options=["Event 1 - 09 November 2025", "Event 2 - 16 November 2025"],
                key="pred_event"
            )
            hadir_sblm_opt = st.selectbox(
                "Hadir di Event Sebelumnya?",
                options=["Tidak Relevan / Event Pertama", "Ya, Pernah Hadir", "Tidak, Belum Pernah Hadir"],
                key="pred_hsblm"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("Prediksi Sekarang", use_container_width=True)

        with col_result:
            p = st.session_state.get('pred_data')
            if p:
                if p['pred'] == 1:
                    st.markdown(
                        '<div class="result-hadir">DIPREDIKSI HADIR</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-tidak-hadir">DIPREDIKSI TIDAK HADIR</div>',
                        unsafe_allow_html=True
                    )

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Peluang Hadir", f"{p['prob_hadir']*100:.1f}%")
                    st.progress(float(p['prob_hadir']))
                with col_p2:
                    st.metric("Peluang Tidak Hadir", f"{p['prob_tidak']*100:.1f}%")
                    st.progress(float(p['prob_tidak']))

                fig_prob = go.Figure(go.Bar(
                    x=["Tidak Hadir", "Hadir"],
                    y=[p['prob_tidak'] * 100, p['prob_hadir'] * 100],
                    marker_color=["#6b9ed4", "#0f2040"],
                    text=[f"{p['prob_tidak']*100:.1f}%", f"{p['prob_hadir']*100:.1f}%"],
                    textposition="outside",
                ))
                fig_prob.update_layout(
                    height=220, showlegend=False,
                    margin=dict(t=10, b=10, l=0, r=0),
                    yaxis=dict(range=[0, 110], showgrid=False, visible=False),
                    xaxis=dict(showgrid=False),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.markdown("""
                <div class="info-box" style="text-align:center;padding:60px 20px;min-height:300px;">
                <div style="font-size:20px;font-weight:800;color:#0f2040;margin-bottom:14px;letter-spacing:1px;">SIAP BERPREDIKSI</div>
                <div style="font-size:17px;color:#4a6fa5;line-height:1.9;">
                Isi form di sebelah kiri<br>dan klik <b style="color:#0f2040;">Prediksi Sekarang</b><br>untuk melihat hasil prediksi.
                </div>
                </div>
                """, unsafe_allow_html=True)

        # ---- Compute and store prediction ----
        if predict_btn:
            jk_enc    = 0 if jk == "Perempuan" else 1
            sp_enc    = 1
            event_enc = 1 if "Event 1" in event else 2
            hsblm_enc = 1 if hadir_sblm_opt == "Ya, Pernah Hadir" else 0

            X_input = pd.DataFrame([{
                "Jenis_Kelamin": jk_enc,
                "Usia": usia,
                "Jarak_km": jarak,
                "Status_Pendaftaran": sp_enc,
                "Event_ID": event_enc,
                "hadir_event_sebelumnya": hsblm_enc,
            }])
            pred_val = int(model.predict(X_input)[0])
            prob_val = model.predict_proba(X_input)[0]

            st.session_state['pred_data'] = {
                'pred': pred_val,
                'prob_hadir': float(prob_val[1]),
                'prob_tidak': float(prob_val[0]),
                'jk': jk, 'jk_enc': jk_enc,
                'usia': usia, 'jarak': jarak,
                'event': event,
                'hadir_sblm_opt': hadir_sblm_opt,
                'hsblm_enc': hsblm_enc,
            }
            st.session_state["pred_log"].append({
                "Waktu": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "Event": event,
                "Jenis Kelamin": jk,
                "Usia": usia,
                "Jarak (km)": round(jarak, 1),
                "Hadir Sebelumnya": hadir_sblm_opt,
                "Prediksi": "Hadir" if pred_val == 1 else "Tidak Hadir",
                "Prob. Hadir (%)": round(float(prob_val[1]) * 100, 1),
                "Prob. Tidak Hadir (%)": round(float(prob_val[0]) * 100, 1),
            })
            _save_log_to_disk(st.session_state["pred_log"])
            st.rerun()

        # ---- Full-width Insight Panel (below both columns) ----
        p = st.session_state.get('pred_data')
        if p:
            pred       = p['pred']
            prob_hadir = p['prob_hadir']
            prob_tidak = p['prob_tidak']
            jk_enc     = p['jk_enc']
            usia       = p['usia']
            jarak      = p['jarak']
            hsblm_enc  = p['hsblm_enc']
            jk         = p['jk']
            event      = p['event']
            hadir_sblm_opt = p['hadir_sblm_opt']

            st.markdown(
                "<hr style='margin:28px 0 16px 0;border:none;border-top:1px solid #e0e0e0;'>",
                unsafe_allow_html=True
            )
            st.markdown("**Analisis Faktor Prediksi**")
            st.markdown("<br>", unsafe_allow_html=True)

            MEAN_JARAK   = 6.96
            MEAN_USIA    = 43.9
            MEDIAN_JARAK = 3.5

            def jarak_insight(km):
                if km <= 3:
                    return "BAIK", "Sangat dekat", f"{km:.1f} km, di bawah median dataset ({MEDIAN_JARAK:.1f} km). Sangat mendukung kehadiran."
                elif km <= 7:
                    return "BAIK", "Dekat", f"{km:.1f} km, di bawah rata-rata dataset ({MEAN_JARAK:.1f} km). Mendukung kehadiran."
                elif km <= 15:
                    return "PERHATIAN", "Di atas rata-rata", f"{km:.1f} km, di atas rata-rata ({MEAN_JARAK:.1f} km). Berpotensi mengurangi motivasi hadir."
                else:
                    return "PERHATIAN", "Jauh", f"{km:.1f} km, jauh di atas rata-rata ({MEAN_JARAK:.1f} km). Faktor risiko dominan (47.73%)."

            def usia_insight(u):
                if u < 35:
                    return "BAIK", "Di bawah rata-rata", f"{u} tahun, lebih muda dari rata-rata ({MEAN_USIA:.1f} thn). Cenderung lebih hadir."
                elif u <= 52:
                    return "BAIK", "Sekitar rata-rata", f"{u} tahun, mendekati rata-rata ({MEAN_USIA:.1f} thn). Tidak menjadi faktor risiko."
                else:
                    return "PERHATIAN", "Di atas rata-rata", f"{u} tahun, lebih tua dari rata-rata ({MEAN_USIA:.1f} thn). Sedikit berisiko."

            def riwayat_insight(h):
                if h == 1:
                    return "BAIK", "Pernah hadir", "Pernah hadir di event sebelumnya. Indikator loyalitas yang baik."
                else:
                    return "NETRAL", "Belum ada riwayat", "Peserta baru atau tidak hadir sebelumnya."

            jk_icon       = "BAIK" if jk_enc == 0 else "NETRAL"
            jk_label_disp = "Perempuan" if jk_enc == 0 else "Laki-Laki"
            jk_note       = "Mayoritas peserta (91.7%) Perempuan. Profil umum." if jk_enc == 0 else "Laki-Laki, minoritas (8.3%) di dataset."

            jarak_icon,   jarak_label,   jarak_note   = jarak_insight(jarak)
            usia_icon,    usia_label,    usia_note    = usia_insight(usia)
            riwayat_icon, riwayat_label, riwayat_note = riwayat_insight(hsblm_enc)

            fc1, fc2, fc3, fc4 = st.columns(4, gap="small")
            factor_cards = [
                (jarak_icon,   "Jarak ke Lokasi", f"{jarak:.1f} km",           jarak_label,   "47.73%", jarak_note),
                (usia_icon,    "Usia",             f"{usia} tahun",             usia_label,    "36.00%", usia_note),
                (riwayat_icon, "Riwayat Hadir",   "Ya" if hsblm_enc == 1 else "Tidak", riwayat_label, "3.35%",  riwayat_note),
                (jk_icon,      "Jenis Kelamin",    jk_label_disp,              "-",           "2.18%",  jk_note),
            ]
            for col_card, (icon, title, value, assessment, imp, note) in zip(
                [fc1, fc2, fc3, fc4], factor_cards
            ):
                with col_card:
                    border_color = "#0f2040" if icon == "BAIK" else ("#b91c1c" if icon == "PERHATIAN" else "#8a9ab5")
                    badge_bg     = "#0f2040" if icon == "BAIK" else ("#b91c1c" if icon == "PERHATIAN" else "#6b7fa0")
                    st.markdown(f"""
                    <div style="border:2px solid {border_color};border-radius:10px;padding:18px;
                                background:#fff;min-height:210px;">
                      <div style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:12px;
                                  font-weight:700;background:{badge_bg};color:#fff;margin-bottom:10px;
                                  letter-spacing:1px;">{icon}</div>
                      <div style="font-size:13px;color:#4a6fa5;text-transform:uppercase;
                                  letter-spacing:0.8px;margin-bottom:6px;font-weight:700;">{title}</div>
                      <div style="font-size:22px;font-weight:800;color:#0f2040;margin-bottom:4px;">{value}</div>
                      <div style="font-size:15px;color:#1e2d4a;margin-bottom:6px;font-weight:600;">{assessment}</div>
                      <div style="font-size:13px;color:#4a6fa5;font-style:italic;margin-bottom:6px;">Importance: {imp}</div>
                      <div style="font-size:14px;color:#555;line-height:1.6;">{note}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Narrative
            dominant_risk     = []
            dominant_positive = []
            if jarak > 15:
                dominant_risk.append(f"jarak yang sangat jauh ({jarak:.1f} km)")
            elif jarak > 7:
                dominant_risk.append(f"jarak di atas rata-rata ({jarak:.1f} km)")
            else:
                dominant_positive.append(f"jarak yang dekat ({jarak:.1f} km)")
            if usia < 35:
                dominant_positive.append("usia yang relatif muda")
            elif usia > 52:
                dominant_risk.append("usia di atas rata-rata peserta")
            else:
                dominant_positive.append("usia dalam rentang normal")
            if hsblm_enc == 1:
                dominant_positive.append("riwayat hadir di event sebelumnya")

            pred_label = "<b>Hadir</b>" if pred == 1 else "<b>Tidak Hadir</b>"
            if dominant_positive and dominant_risk:
                if pred == 1:
                    narrative = (
                        f"Model memprediksi {pred_label} karena faktor positif "
                        f"({', '.join(dominant_positive)}) lebih dominan dibanding "
                        f"faktor risiko ({', '.join(dominant_risk)})."
                    )
                else:
                    narrative = (
                        f"Model memprediksi {pred_label} karena faktor risiko "
                        f"({', '.join(dominant_risk)}) tidak bisa dikompensasi oleh "
                        f"faktor positif ({', '.join(dominant_positive)})."
                    )
            elif dominant_positive:
                narrative = (
                    f"Model memprediksi {pred_label} karena kombinasi faktor positif: "
                    f"{', '.join(dominant_positive)}."
                )
            else:
                narrative = (
                    f"Model memprediksi {pred_label} terutama karena: "
                    f"{', '.join(dominant_risk)}."
                )

            st.markdown(
                f"<div class='info-box' style='margin-top:16px;font-size:13px;'>"
                f"<b>Kesimpulan:</b> {narrative}"
                f"</div>",
                unsafe_allow_html=True
            )

            # CSV download
            result_dict = {
                "Jenis Kelamin":                [jk],
                "Usia":                         [usia],
                "Jarak (km)":                   [jarak],
                "Event":                        [event],
                "Hadir Event Sebelumnya":       [hadir_sblm_opt],
                "Prediksi":                     ["Hadir" if pred == 1 else "Tidak Hadir"],
                "Probabilitas Hadir (%)":       [round(prob_hadir * 100, 2)],
                "Probabilitas Tidak Hadir (%)": [round(prob_tidak * 100, 2)],
            }
            df_result = pd.DataFrame(result_dict)
            csv_bytes = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="Unduh Hasil Prediksi (CSV)",
                data=csv_bytes,
                file_name="hasil_prediksi.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except FileNotFoundError:
        st.error("Model tidak ditemukan. Jalankan `python scripts/02_train.py` terlebih dahulu.")



# ============================================================
# TAB 3 Ã¢â‚¬â€ EVALUASI MODEL
# ============================================================
with tab_eval:
    st.markdown('<div class="section-header">Evaluasi Model Random Forest</div>', unsafe_allow_html=True)

    try:
        m = load_metrics()

        # ---- Metric Cards ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi", f"{m['accuracy_pct']:.2f}%")
        c2.metric("Presisi (Weighted)", f"{m['precision_weighted']*100:.2f}%")
        c3.metric("Recall (Weighted)", f"{m['recall_weighted']*100:.2f}%")
        c4.metric("F1-Score (Weighted)", f"{m['f1_weighted']*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Per-class table ----
        st.markdown("**Metrik Per Kelas**")
        pc = m["per_class"]
        table_html = f"""
        <table class="metric-table">
            <tr>
                <th>Kelas</th>
                <th>Presisi</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
            <tr>
                <td>Hadir</td>
                <td>{pc['Hadir']['precision']*100:.2f}%</td>
                <td>{pc['Hadir']['recall']*100:.2f}%</td>
                <td>{pc['Hadir']['f1']*100:.2f}%</td>
                <td>{pc['Hadir']['support']}</td>
            </tr>
            <tr>
                <td>Tidak Hadir</td>
                <td>{pc['Tidak Hadir']['precision']*100:.2f}%</td>
                <td>{pc['Tidak Hadir']['recall']*100:.2f}%</td>
                <td>{pc['Tidak Hadir']['f1']*100:.2f}%</td>
                <td>{pc['Tidak Hadir']['support']}</td>
            </tr>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Charts ----
        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            st.image(IMG_CM, use_container_width=True)
            cm_data = m["confusion_matrix"]
            st.markdown(
                f"<div class='info-box' style='font-size:13px;'>"
                f"<b>True Positive (TP):</b> {cm_data['TP']} Ã¢â‚¬â€ Hadir, diprediksi Hadir<br>"
                f"<b>True Negative (TN):</b> {cm_data['TN']} Ã¢â‚¬â€ Tidak Hadir, diprediksi Tidak Hadir<br>"
                f"<b>False Positive (FP):</b> {cm_data['FP']} Ã¢â‚¬â€ Tidak Hadir, diprediksi Hadir<br>"
                f"<b>False Negative (FN):</b> {cm_data['FN']} Ã¢â‚¬â€ Hadir, diprediksi Tidak Hadir"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_fi:
            st.markdown("**Feature Importance**")
            st.image(IMG_FI, use_container_width=True)

            # Feature importance table
            fi_sorted = m["feature_importance_sorted"]
            feat_labels = {
                "Jenis_Kelamin": "Jenis Kelamin",
                "Usia": "Usia",
                "Jarak_km": "Jarak (km)",
                "Status_Pendaftaran": "Status Pendaftaran",
                "Event_ID": "Event ID",
                "hadir_event_sebelumnya": "Hadir Event Sebelumnya",
            }
            fi_html = '<table class="metric-table"><tr><th>Fitur</th><th>Importance</th><th>Peringkat</th></tr>'
            for rank, item in enumerate(fi_sorted, 1):
                fi_html += f"<tr><td>{feat_labels.get(item['feature'], item['feature'])}</td><td>{item['importance']*100:.2f}%</td><td>#{rank}</td></tr>"
            fi_html += "</table>"
            st.markdown(fi_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Model Info ----
        st.markdown("**Informasi Model**")
        mp = m["model_params"]
        info_html = f"""
        <div class="info-box">
        <b>Algoritma:</b> Random Forest Classifier (scikit-learn) &nbsp;|&nbsp;
        <b>n_estimators:</b> {mp['n_estimators']} &nbsp;|&nbsp;
        <b>class_weight:</b> {mp['class_weight']} &nbsp;|&nbsp;
        <b>random_state:</b> {mp['random_state']} &nbsp;|&nbsp;
        <b>Train/Test Split:</b> {int((1-mp['test_size'])*100)}/{int(mp['test_size']*100)} &nbsp;|&nbsp;
        <b>Data Training:</b> {m['n_train']} baris &nbsp;|&nbsp;
        <b>Data Test:</b> {m['n_test']} baris
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

        # ---- Interpretasi & Temuan ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Interpretasi &amp; Temuan Model</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 3 Finding Cards ---
        fi_top = m["feature_importance_sorted"]
        top1_label = {"Jenis_Kelamin": "Jenis Kelamin", "Usia": "Usia", "Jarak_km": "Jarak (km)",
                      "Status_Pendaftaran": "Status Pendaftaran", "Event_ID": "Event ID",
                      "hadir_event_sebelumnya": "Hadir Event Sebelumnya"}.get(fi_top[0]["feature"], fi_top[0]["feature"])
        top2_label = {"Jenis_Kelamin": "Jenis Kelamin", "Usia": "Usia", "Jarak_km": "Jarak (km)",
                      "Status_Pendaftaran": "Status Pendaftaran", "Event_ID": "Event ID",
                      "hadir_event_sebelumnya": "Hadir Event Sebelumnya"}.get(fi_top[1]["feature"], fi_top[1]["feature"])
        cm_d = m["confusion_matrix"]
        pct_hadir = m["distribusi_dataset"]["hadir_pct"]

        card1, card2, card3 = st.columns(3, gap="medium")
        with card1:
            st.markdown(f"""
            <div class="info-box" style="border-left:4px solid #111;padding-left:16px;">
            <div style="font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;font-weight:600;">Pola Dominan</div>
            <div style="font-size:28px;font-weight:700;color:#111;">{pct_hadir:.0f}%</div>
            <div style="font-size:14px;color:#444;margin-top:4px;">peserta hadir</div>
            <div style="font-size:14px;color:#555;margin-top:10px;line-height:1.6;">
            Dataset didominasi oleh peserta yang hadir. Model dilatih dengan <code>class_weight='balanced'</code>
            untuk mengimbangi ketimpangan ini.
            </div>
            </div>""", unsafe_allow_html=True)
        with card2:
            st.markdown(f"""
            <div class="info-box" style="border-left:4px solid #111;padding-left:16px;">
            <div style="font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;font-weight:600;">Fitur Paling Berpengaruh</div>
            <div style="font-size:22px;font-weight:700;color:#111;">{top1_label}</div>
            <div style="font-size:14px;color:#444;margin-top:4px;">importance: {fi_top[0]['importance']*100:.1f}%</div>
            <div style="font-size:14px;color:#555;margin-top:10px;line-height:1.6;">
            Diikuti oleh <b>{top2_label}</b> ({fi_top[1]['importance']*100:.1f}%).
            Kedua fitur ini menjelaskan <b>{(fi_top[0]['importance']+fi_top[1]['importance'])*100:.1f}%</b> dari keputusan model.
            </div>
            </div>""", unsafe_allow_html=True)
        with card3:
            st.markdown(f"""
            <div class="info-box" style="border-left:4px solid #111;padding-left:16px;">
            <div style="font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;font-weight:600;">Ketepatan Prediksi</div>
            <div style="font-size:28px;font-weight:700;color:#111;">{m['accuracy_pct']:.0f}%</div>
            <div style="font-size:14px;color:#444;margin-top:4px;">akurasi keseluruhan</div>
            <div style="font-size:14px;color:#555;margin-top:10px;line-height:1.6;">
            Dari {m['n_test']} data uji: <b>{cm_d['TP']+cm_d['TN']} prediksi benar</b>,
            {cm_d['FP']} false alarm, {cm_d['FN']} yang terlewat.
            </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Feature Narrative ---
        st.markdown("**Analisis Fitur Ã¢â‚¬â€ Apa yang Model Pelajari**")

        feat_label_map = {
            "Jenis_Kelamin": "Jenis Kelamin",
            "Usia": "Usia",
            "Jarak_km": "Jarak (km)",
            "Status_Pendaftaran": "Status Pendaftaran",
            "Event_ID": "Event ID",
            "hadir_event_sebelumnya": "Hadir Event Sebelumnya",
        }
        feat_narratives = {
            "Jarak_km": (
                "Ini faktor paling berpengaruh dalam model. <b>Semakin dekat rumah peserta ke lokasi event, "
                "semakin tinggi kemungkinan mereka datang</b>. Peserta yang tinggal di bawah 5 km dari "
                "Halim Perdanakusuma punya tingkat kehadiran jauh lebih tinggi. "
                "Wajar saja, jarak yang jauh membuat perjalanan lebih melelahkan dan mahal."
            ),
            "Usia": (
                "Faktor terbesar kedua. <b>Peserta usia 30 sampai 50 tahun paling konsisten hadir</b>, "
                "dibanding yang lebih muda dari 28 tahun atau lebih tua dari 58 tahun. "
                "Usia produktif biasanya punya jadwal yang lebih teratur dan motivasi lebih tinggi untuk belajar keterampilan baru."
            ),
            "Status_Pendaftaran": (
                "Fitur ketiga. <b>Peserta yang sudah terdaftar resmi lebih cenderung datang</b> "
                "dibanding yang belum terdaftar formal. "
                "Masuk akal, karena mendaftar menunjukkan niat yang lebih serius sejak awal."
            ),
            "Event_ID": (
                "Event mana yang diikuti (Event 1 atau 2) cukup berpengaruh. "
                "<b>Pola kehadiran berbeda di setiap event</b>, "
                "kemungkinan karena perbedaan hari pelaksanaan, cara promosi, "
                "atau seberapa luas informasi acara tersebar sebelumnya."
            ),
            "hadir_event_sebelumnya": (
                "Pengaruhnya kecil (3.35%), tapi bermakna: "
                "<b>peserta yang pernah hadir sebelumnya cenderung hadir lagi</b>. "
                "Ini menunjukkan mereka punya pengalaman positif dan loyal terhadap kegiatan pelatihan PT CLN."
            ),
            "Jenis_Kelamin": (
                "Pengaruhnya paling kecil (2.18%). Karena 91% peserta adalah perempuan, "
                "model tidak banyak belajar dari perbedaan gender. "
                "<b>Jenis kelamin bukan faktor penentu kehadiran</b> dalam pelatihan ini."
            ),
        }

        for item in fi_top:
            feat_key = item["feature"]
            feat_name = feat_label_map.get(feat_key, feat_key)
            imp_pct = item["importance"] * 100
            bar_width = int(imp_pct * 1.8)  # scale for visual
            narrative = feat_narratives.get(feat_key, "Ã¢â‚¬â€")
            rank_num = fi_top.index(item) + 1

            st.markdown(f"""
            <div class="info-box" style="margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:14px;font-weight:700;color:#4a6fa5;background:#eef1f8;padding:3px 10px;border-radius:4px;">#{rank_num}</span>
                <span style="font-weight:700;font-size:15px;color:#111;">{feat_name}</span>
                <span style="margin-left:auto;font-size:13px;font-weight:600;color:#333;">{imp_pct:.2f}%</span>
              </div>
              <div style="background:#d0daea;border-radius:4px;height:6px;margin-bottom:10px;">
                <div style="background:#0f2040;border-radius:4px;height:6px;width:{min(bar_width,100)}%;"></div>
              </div>
              <div style="font-size:13px;color:#444;line-height:1.6;">{narrative}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Overall Assessment ---
        st.markdown("**Penilaian Keseluruhan Model**")
        hadir_f1 = m["per_class"]["Hadir"]["f1"] * 100
        tidak_f1 = m["per_class"]["Tidak Hadir"]["f1"] * 100
        st.markdown(f"""
        <div class="info-box" style="line-height:1.8;font-size:15px;">
        <p>Model ini dilatih dari <b>{m['n_total']} data</b> ({m['n_train']} untuk training, {m['n_test']} untuk testing),
        dan berhasil menangkap pola kehadiran peserta dengan akurasi <b>{m['accuracy_pct']:.2f}%</b>.
        Model bekerja sangat baik untuk mengidentifikasi peserta yang kemungkinan <b>Hadir</b> (F1-Score: {hadir_f1:.1f}%),
        namun kurang optimal untuk mendeteksi yang <b>Tidak Hadir</b> (F1-Score: {tidak_f1:.1f}%).</p>

        <p>Keterbatasan ini bukan masalah teknis. Ini terjadi karena data tidak seimbang:
        {pct_hadir:.1f}% peserta hadir vs {100-pct_hadir:.1f}% tidak hadir.
        Konfigurasi <code>class_weight='balanced'</code> sudah digunakan untuk mengurangi bias ini,
        tapi dengan hanya 300 baris data, sinyal dari kelas minoritas masih terbatas.</p>

        <p><b>Rekomendasi:</b> Gunakan model ini sebagai alat bantu awal.
        Fokuskan perhatian pada peserta yang diprediksi <b>Tidak Hadir</b> untuk ditindaklanjuti
        (pengingat atau konfirmasi ulang). Jangan jadikan ini sebagai keputusan akhir.
        Performa model akan meningkat seiring bertambahnya data dari event berikutnya.</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- PDF Download ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Unduh Laporan Evaluasi**")

        def generate_pdf(metrics_data):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Laporan Evaluasi Model Random Forest", ln=True, align="C")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, "Prediksi Kehadiran Anggota - PT Cahaya Ladara Nusantara", ln=True, align="C")
            pdf.ln(6)

            # Garis pemisah
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(6)

            # Ringkasan Metrik
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "1. Ringkasan Metrik Evaluasi", ln=True)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(30, 30, 30)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(60, 8, "Metrik", border=1, fill=True)
            pdf.cell(60, 8, "Nilai", border=1, fill=True)
            pdf.cell(70, 8, "Keterangan", border=1, fill=True, ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            rows_metrics = [
                ("Akurasi", f"{metrics_data['accuracy_pct']:.2f}%", "Proporsi prediksi benar"),
                ("Presisi (Weighted)", f"{metrics_data['precision_weighted']*100:.2f}%", "Ketepatan prediksi positif"),
                ("Recall (Weighted)", f"{metrics_data['recall_weighted']*100:.2f}%", "Kemampuan deteksi positif"),
                ("F1-Score (Weighted)", f"{metrics_data['f1_weighted']*100:.2f}%", "Harmonic mean P & R"),
                ("Data Training", str(metrics_data["n_train"]) + " baris", "80% dari total dataset"),
                ("Data Test", str(metrics_data["n_test"]) + " baris", "20% dari total dataset"),
                ("Total Dataset", str(metrics_data["n_total"]) + " baris", "2 event digabungkan"),
            ]
            fill = False
            for r in rows_metrics:
                pdf.set_fill_color(248, 248, 248) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(60, 8, r[0], border=1, fill=fill)
                pdf.cell(60, 8, r[1], border=1, fill=fill)
                pdf.cell(70, 8, r[2], border=1, fill=fill, ln=True)
                fill = not fill
            pdf.ln(6)

            # Per-class
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "2. Metrik Per Kelas", ln=True)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(30, 30, 30)
            pdf.set_text_color(255, 255, 255)
            for h in ["Kelas", "Presisi", "Recall", "F1-Score", "Support"]:
                pdf.cell(38, 8, h, border=1, fill=True)
            pdf.ln()

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            pc = metrics_data["per_class"]
            for kelas, val in pc.items():
                pdf.cell(38, 8, kelas, border=1)
                pdf.cell(38, 8, f"{val['precision']*100:.2f}%", border=1)
                pdf.cell(38, 8, f"{val['recall']*100:.2f}%", border=1)
                pdf.cell(38, 8, f"{val['f1']*100:.2f}%", border=1)
                pdf.cell(38, 8, str(val['support']), border=1, ln=True)
            pdf.ln(6)

            # Confusion Matrix
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "3. Confusion Matrix", ln=True)
            pdf.ln(2)
            cm_d = metrics_data["confusion_matrix"]
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"True Positive (TP)  = {cm_d['TP']}  - Hadir, diprediksi Hadir", ln=True)
            pdf.cell(0, 7, f"True Negative (TN)  = {cm_d['TN']}  - Tidak Hadir, diprediksi Tidak Hadir", ln=True)
            pdf.cell(0, 7, f"False Positive (FP) = {cm_d['FP']} - Tidak Hadir, diprediksi Hadir", ln=True)
            pdf.cell(0, 7, f"False Negative (FN) = {cm_d['FN']}  - Hadir, diprediksi Tidak Hadir", ln=True)
            pdf.ln(4)

            # Feature Importance
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "4. Feature Importance", ln=True)
            pdf.ln(2)

            feat_labels_pdf = {
                "Jenis_Kelamin": "Jenis Kelamin",
                "Usia": "Usia",
                "Jarak_km": "Jarak (km)",
                "Status_Pendaftaran": "Status Pendaftaran",
                "Event_ID": "Event ID",
                "hadir_event_sebelumnya": "Hadir Event Sebelumnya",
            }
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(30, 30, 30)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(100, 8, "Fitur", border=1, fill=True)
            pdf.cell(45, 8, "Importance", border=1, fill=True)
            pdf.cell(45, 8, "Peringkat", border=1, fill=True, ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            fill = False
            for rank, item in enumerate(metrics_data["feature_importance_sorted"], 1):
                pdf.set_fill_color(248, 248, 248) if fill else pdf.set_fill_color(255, 255, 255)
                label = feat_labels_pdf.get(item["feature"], item["feature"])
                pdf.cell(100, 8, label, border=1, fill=fill)
                pdf.cell(45, 8, f"{item['importance']*100:.2f}%", border=1, fill=fill)
                pdf.cell(45, 8, f"#{rank}", border=1, fill=fill, ln=True)
                fill = not fill
            pdf.ln(6)

            # Model Info
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "5. Konfigurasi Model", ln=True)
            pdf.ln(2)
            mp = metrics_data["model_params"]
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"Algoritma      : Random Forest Classifier (scikit-learn)", ln=True)
            pdf.cell(0, 7, f"n_estimators   : {mp['n_estimators']} pohon keputusan", ln=True)
            pdf.cell(0, 7, f"class_weight   : {mp['class_weight']}", ln=True)
            pdf.cell(0, 7, f"random_state   : {mp['random_state']}", ln=True)
            pdf.cell(0, 7, f"Train/Test     : {int((1-mp['test_size'])*100)}/{int(mp['test_size']*100)}", ln=True)

            pdf_bytes = bytes(pdf.output())
            return pdf_bytes

        if st.button("Generate & Unduh Laporan PDF", use_container_width=True):
            with st.spinner("Membuat laporan PDF..."):
                pdf_bytes = generate_pdf(m)
            st.download_button(
                label="Unduh Laporan PDF",
                data=pdf_bytes,
                file_name="laporan_evaluasi_model_rf.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("Laporan PDF berhasil dibuat!")

    except FileNotFoundError:
        st.error("Hasil evaluasi tidak ditemukan. Jalankan `python scripts/03_evaluate.py` terlebih dahulu.")


# ============================================================
# TAB 4 Ã¢â‚¬â€ LOG PREDIKSI
# ============================================================
with tab_log:
    header_title = "Log Prediksi" if IS_LOCAL else "Log Prediksi Sesi Ini"
    st.markdown(f'<div class="section-header">{header_title}</div>', unsafe_allow_html=True)
    if IS_LOCAL:
        st.markdown("<p style='color:#4a6fa5;font-size:15px;'>Log tersimpan secara permanen di mesin ini dan akan tetap ada setelah aplikasi di-restart.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    log = st.session_state.get("pred_log", [])

    if not log:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:48px 20px;">
        <div style="font-size:20px;font-weight:800;color:#0f2040;margin-bottom:10px;">Belum ada prediksi</div>
        <div style="font-size:16px;color:#4a6fa5;">Lakukan prediksi di tab Prediksi Kehadiran<br>untuk melihat riwayat di sini.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total     = len(log)
        hadir_cnt = sum(1 for e in log if e["Prediksi"] == "Hadir")
        tidak_cnt = total - hadir_cnt

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Prediksi", total)
        c2.metric("Diprediksi Hadir", hadir_cnt)
        c3.metric("Diprediksi Tidak Hadir", tidak_cnt)

        st.markdown("<br>", unsafe_allow_html=True)

        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            csv_log = df_log.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="Unduh Log Prediksi (CSV)",
                data=csv_log,
                file_name="log_prediksi.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("Hapus Log", use_container_width=True):
                st.session_state["pred_log"] = []
                _save_log_to_disk([])  # clear disk file on localhost too
                st.rerun()
