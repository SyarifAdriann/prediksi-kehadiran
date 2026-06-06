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
import joblib
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
    page_icon="📋",
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
IMG_DK = os.path.join(ROOT, "outputs", "distribusi_kehadiran.png")

FEATURE_COLS = [
    "Jenis_Kelamin", "Usia", "Jarak_km",
    "Status_Pendaftaran", "Event_ID", "hadir_event_sebelumnya",
]

# ============================================================
# CSS — MONOCHROME CLEAN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
}

/* ── Background ── */
.stApp {
    background-color: #f9f9f9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111;
    color: #ffffff;
}
[data-testid="stSidebar"] * {
    color: #cccccc !important;
    font-size: 15px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 18px !important;
}

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #e0e0e0;
}
[data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: #888888;
    font-weight: 500;
    padding: 10px 22px;
    font-size: 16px;
}
[aria-selected="true"] {
    color: #111111 !important;
    border-bottom: 2px solid #111111 !important;
    background: transparent !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 14px !important;
    color: #666666;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700;
    color: #111111;
}

/* ── Form / widget labels ── */
[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label,
.stRadio label,
.stCheckbox label {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #222222 !important;
    margin-bottom: 4px !important;
}
/* Selectbox and input text */
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-baseweb="input"] input {
    font-size: 15px !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #111111;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 22px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #333333;
    color: #ffffff;
}

/* ── Download button ── */
.stDownloadButton > button {
    background-color: #ffffff;
    color: #111111;
    border: 1px solid #cccccc;
    border-radius: 6px;
    font-weight: 600;
    font-size: 15px;
}
.stDownloadButton > button:hover {
    background-color: #f0f0f0;
}

/* ── Prediksi result boxes ── */
.result-hadir {
    background: #111111;
    color: #ffffff;
    border-radius: 10px;
    padding: 26px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 16px 0;
}
.result-tidak-hadir {
    background: #ffffff;
    color: #111111;
    border: 2px solid #111111;
    border-radius: 10px;
    padding: 26px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 16px 0;
}

/* ── Section header ── */
.section-header {
    font-size: 26px;
    font-weight: 700;
    color: #111111;
    margin-bottom: 4px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e8e8e8;
}

/* ── Info box ── */
.info-box {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 8px 0;
    font-size: 15px;
    color: #333333;
    line-height: 1.65;
}

/* ── Metric table ── */
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
}
.metric-table th {
    background: #111111;
    color: #ffffff;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
}
.metric-table td {
    padding: 11px 16px;
    border-bottom: 1px solid #f0f0f0;
    color: #333333;
    font-size: 14px;
    line-height: 1.5;
}
.metric-table tr:nth-child(even) td {
    background: #f8f8f8;
}

/* ── General markdown / paragraph text ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 15px;
    line-height: 1.65;
}
[data-testid="stMarkdownContainer"] b,
[data-testid="stMarkdownContainer"] strong {
    font-weight: 700;
}

/* ── Hide streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Input widget visible borders ── */
div[data-baseweb="select"] > div {
    border: 1.5px solid #cccccc !important;
    border-radius: 6px !important;
    background-color: #ffffff !important;
}
div[data-baseweb="input"] > div {
    border: 1.5px solid #cccccc !important;
    border-radius: 6px !important;
    background-color: #ffffff !important;
}
div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="input"]:focus-within > div {
    border: 1.5px solid #111111 !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.07) !important;
}
</style>
""", unsafe_allow_html=True)


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


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📋 Prediksi Kehadiran")
    st.markdown("**PT Cahaya Ladara Nusantara**")
    st.markdown("---")
    st.markdown("""
**Navigasi Tab:**
- 🔮 &nbsp;Prediksi Kehadiran
- 📊 &nbsp;Dashboard
- 📈 &nbsp;Evaluasi Model
""")
    st.markdown("---")
    st.markdown("""
<div style="font-size:12px; color:#888;">
Pelatihan Makanan Viral<br>
Sushi & Onigiri<br>
Event 1: 09 November 2025<br>
Event 2: 16 November 2025
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    try:
        m = load_metrics()
        st.markdown(f"<div style='font-size:12px;color:#888;'>Model Accuracy<br><b style='font-size:22px;color:#fff;'>{m['accuracy_pct']:.1f}%</b></div>", unsafe_allow_html=True)
    except Exception:
        pass


# ============================================================
# JUDUL
# ============================================================
st.markdown("# Sistem Prediksi Kehadiran Anggota")
st.markdown(
    "<p style='color:#666;font-size:14px;margin-top:-12px;'>"
    "Algoritma Random Forest — Data Riwayat Pelatihan Sushi & Onigiri, PT Cahaya Ladara Nusantara"
    "</p>",
    unsafe_allow_html=True
)

tab_pred, tab_dash, tab_eval = st.tabs(["🔮  Prediksi Kehadiran", "📊  Dashboard", "📈  Evaluasi Model"])

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
                marker_color="#1a1a1a", text=hadir_vals,
                textposition="outside"
            ))
            fig_ev.add_trace(go.Bar(
                name="Tidak Hadir", x=ev_labels, y=tidak_vals,
                marker_color="#cccccc", text=tidak_vals,
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
                marker=dict(colors=["#1a1a1a", "#aaaaaa"]),
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
                marker_color="#555555", marker_line_color="#ffffff",
                marker_line_width=1,
                name="Usia"
            ))
            fig_usia.add_vline(
                x=df["Usia"].mean(), line_dash="dash",
                line_color="#111111", line_width=2,
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
                marker_color="#555555", marker_line_color="#ffffff",
                marker_line_width=1,
                name="Jarak"
            ))
            fig_jarak.add_vline(
                x=df["Jarak_km"].median(), line_dash="dash",
                line_color="#111111", line_width=2,
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
        "<p style='color:#666;font-size:13px;'>Masukkan data peserta untuk mendapatkan prediksi kehadiran beserta analisis faktor yang memengaruhinya.</p>",
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
            st.markdown(
                "<div style='background:#f9f9f9;border:1.5px solid #e0e0e0;border-radius:10px;padding:20px 20px 8px 20px;'>",
                unsafe_allow_html=True
            )
            st.markdown("**📋 Data Peserta**")
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
                options=["Event 1 \u2014 09 November 2025", "Event 2 \u2014 16 November 2025"],
                key="pred_event"
            )
            hadir_sblm_opt = st.selectbox(
                "Hadir di Event Sebelumnya?",
                options=["Tidak Relevan / Event Pertama", "Ya \u2014 Hadir", "Tidak \u2014 Tidak Hadir"],
                key="pred_hsblm"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("\U0001f52e Prediksi Sekarang", use_container_width=True)

        with col_result:
            p = st.session_state.get('pred_data')
            if p:
                if p['pred'] == 1:
                    st.markdown(
                        '<div class="result-hadir">\u2713 DIPREDIKSI HADIR</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-tidak-hadir">\u2717 DIPREDIKSI TIDAK HADIR</div>',
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
                    marker_color=["#cccccc", "#1a1a1a"],
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
                <div class="info-box" style="text-align:center;padding:60px 20px;color:#999;min-height:300px;">
                <div style="font-size:40px;margin-bottom:12px;">&#128302;</div>
                Isi form di sebelah kiri<br>dan klik <b style="color:#333;">Prediksi Sekarang</b><br>untuk melihat hasil prediksi.
                </div>
                """, unsafe_allow_html=True)

        # ---- Compute and store prediction ----
        if predict_btn:
            jk_enc    = 0 if jk == "Perempuan" else 1
            sp_enc    = 1
            event_enc = 1 if "Event 1" in event else 2
            hsblm_enc = 1 if hadir_sblm_opt == "Ya \u2014 Hadir" else 0

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
                    return "\u2705", "Sangat dekat", f"{km:.1f} km - di bawah median dataset ({MEDIAN_JARAK:.1f} km). Sangat mendukung kehadiran."
                elif km <= 7:
                    return "\u2705", "Dekat", f"{km:.1f} km - di bawah rata-rata dataset ({MEAN_JARAK:.1f} km). Mendukung kehadiran."
                elif km <= 15:
                    return "\u26a0\ufe0f", "Di atas rata-rata", f"{km:.1f} km - di atas rata-rata ({MEAN_JARAK:.1f} km). Berpotensi mengurangi motivasi hadir."
                else:
                    return "\u26a0\ufe0f", "Jauh", f"{km:.1f} km - jauh di atas rata-rata ({MEAN_JARAK:.1f} km). Faktor risiko dominan (47.73%)."

            def usia_insight(u):
                if u < 35:
                    return "\u2705", "Di bawah rata-rata", f"{u} tahun - lebih muda dari rata-rata ({MEAN_USIA:.1f} thn). Cenderung lebih hadir."
                elif u <= 52:
                    return "\u2705", "Sekitar rata-rata", f"{u} tahun - mendekati rata-rata ({MEAN_USIA:.1f} thn). Tidak menjadi faktor risiko."
                else:
                    return "\u26a0\ufe0f", "Di atas rata-rata", f"{u} tahun - lebih tua dari rata-rata ({MEAN_USIA:.1f} thn). Sedikit berisiko."

            def riwayat_insight(h):
                if h == 1:
                    return "\u2705", "Pernah hadir", "Memiliki riwayat hadir di event sebelumnya - indikator loyalitas."
                else:
                    return "\u2796", "Belum ada riwayat", "Peserta baru atau tidak hadir sebelumnya."

            jk_icon       = "\u2705" if jk_enc == 0 else "\u2796"
            jk_label_disp = "Perempuan" if jk_enc == 0 else "Laki-Laki"
            jk_note       = "Mayoritas peserta (91.7%) Perempuan - profil umum." if jk_enc == 0 else "Laki-Laki minoritas (8.3%) di dataset."

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
                    border_color = "#27ae60" if icon == "\u2705" else ("#e67e22" if icon == "\u26a0\ufe0f" else "#aaaaaa")
                    st.markdown(f"""
                    <div style="border:1.5px solid {border_color};border-radius:8px;padding:16px;
                                background:#fff;min-height:200px;">
                      <div style="font-size:24px;margin-bottom:8px;">{icon}</div>
                      <div style="font-size:13px;color:#888;text-transform:uppercase;
                                  letter-spacing:0.5px;margin-bottom:6px;font-weight:600;">{title}</div>
                      <div style="font-size:19px;font-weight:700;color:#111;margin-bottom:4px;">{value}</div>
                      <div style="font-size:14px;color:#444;margin-bottom:6px;">{assessment}</div>
                      <div style="font-size:12px;color:#aaa;font-style:italic;margin-bottom:6px;">Importance: {imp}</div>
                      <div style="font-size:13px;color:#555;line-height:1.5;">{note}</div>
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
                label="\u2b07\ufe0f Unduh Hasil Prediksi (CSV)",
                data=csv_bytes,
                file_name="hasil_prediksi.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except FileNotFoundError:
        st.error("Model tidak ditemukan. Jalankan `python scripts/02_train.py` terlebih dahulu.")



# ============================================================
# TAB 3 — EVALUASI MODEL
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
                f"<b>True Positive (TP):</b> {cm_data['TP']} — Hadir, diprediksi Hadir<br>"
                f"<b>True Negative (TN):</b> {cm_data['TN']} — Tidak Hadir, diprediksi Tidak Hadir<br>"
                f"<b>False Positive (FP):</b> {cm_data['FP']} — Tidak Hadir, diprediksi Hadir<br>"
                f"<b>False Negative (FN):</b> {cm_data['FN']} — Hadir, diprediksi Tidak Hadir"
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
        st.markdown("**Analisis Fitur — Apa yang Model Pelajari**")

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
                "Faktor paling dominan. Model menemukan bahwa <b>semakin dekat tempat tinggal peserta ke lokasi event, "
                "semakin tinggi kemungkinan hadir</b>. Peserta yang tinggal di radius &lt;5 km dari Halim Perdanakusuma "
                "memiliki kecenderungan hadir yang jauh lebih tinggi. Ini konsisten secara logis — jarak yang jauh "
                "meningkatkan biaya waktu dan transportasi, yang menjadi penghalang kehadiran."
            ),
            "Usia": (
                "Faktor terbesar kedua. Pola yang ditemukan: <b>peserta berusia 30–50 tahun cenderung lebih konsisten hadir</b> "
                "dibanding peserta yang sangat muda (&lt;28 tahun) atau sangat tua (&gt;58 tahun). "
                "Rentang usia produktif ini kemungkinan besar berkaitan dengan stabilitas jadwal dan motivasi pengembangan keterampilan."
            ),
            "Status_Pendaftaran": (
                "Fitur ketiga yang berpengaruh. Peserta yang berstatus <b>Terdaftar resmi cenderung lebih berkomitmen untuk hadir</b> "
                "dibanding peserta yang tidak terdaftar secara formal. Status pendaftaran mencerminkan tingkat niat awal peserta."
            ),
            "Event_ID": (
                "Identitas event (Event 1 vs Event 2) berkontribusi cukup signifikan. Ini mengindikasikan bahwa "
                "<b>ada perbedaan pola kehadiran antar event</b>, kemungkinan dipengaruhi oleh hari, promosi, atau "
                "informasi yang disebarkan sebelum event berlangsung."
            ),
            "hadir_event_sebelumnya": (
                "Meskipun memiliki importance relatif kecil (3.35%), fitur ini bermakna secara kontekstual: "
                "<b>peserta yang pernah hadir di event sebelumnya cenderung hadir kembali</b>. "
                "Ini adalah indikator loyalitas dan kepuasan peserta terhadap kegiatan pelatihan PT CLN."
            ),
            "Jenis_Kelamin": (
                "Fitur dengan kontribusi terkecil (2.18%). Dataset didominasi oleh peserta Perempuan (91%), "
                "sehingga pola antar gender sulit dibedakan secara statistik. "
                "<b>Jenis kelamin bukan penentu utama kehadiran</b> dalam konteks pelatihan kuliner ini."
            ),
        }

        for item in fi_top:
            feat_key = item["feature"]
            feat_name = feat_label_map.get(feat_key, feat_key)
            imp_pct = item["importance"] * 100
            bar_width = int(imp_pct * 1.8)  # scale for visual
            narrative = feat_narratives.get(feat_key, "—")
            rank_badge = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][fi_top.index(item)]

            st.markdown(f"""
            <div class="info-box" style="margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:18px;">{rank_badge}</span>
                <span style="font-weight:700;font-size:15px;color:#111;">{feat_name}</span>
                <span style="margin-left:auto;font-size:13px;font-weight:600;color:#333;">{imp_pct:.2f}%</span>
              </div>
              <div style="background:#f0f0f0;border-radius:4px;height:6px;margin-bottom:10px;">
                <div style="background:#111;border-radius:4px;height:6px;width:{min(bar_width,100)}%;"></div>
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
        <div class="info-box" style="line-height:1.8;font-size:13px;">
        <p>Model Random Forest yang dilatih dari <b>{m['n_total']} data</b> ({m['n_train']} training, {m['n_test']} testing)
        berhasil menangkap pola utama kehadiran peserta dengan akurasi <b>{m['accuracy_pct']:.2f}%</b>.
        Model sangat baik dalam mendeteksi peserta yang akan <b>Hadir</b> (F1-Score: {hadir_f1:.1f}%),
        namun menunjukkan keterbatasan dalam mendeteksi peserta yang <b>Tidak Hadir</b> (F1-Score: {tidak_f1:.1f}%).</p>

        <p>Keterbatasan ini bukan bug — ini adalah konsekuensi wajar dari distribusi data yang tidak seimbang
        ({pct_hadir:.1f}% Hadir vs {100-pct_hadir:.1f}% Tidak Hadir). Model telah dikonfigurasi dengan
        <code>class_weight='balanced'</code> untuk memitigasi hal ini, namun dengan dataset 300 baris,
        sinyal untuk kelas minoritas masih terbatas.</p>

        <p><b>Rekomendasi:</b> Untuk prediksi operasional, gunakan model ini sebagai <i>screening tool</i>
        — fokus pada peserta yang diprediksi <b>Tidak Hadir</b> sebagai target tindak lanjut (reminder, konfirmasi ulang),
        bukan sebagai keputusan final. Akurasi model akan meningkat seiring bertambahnya data historis event berikutnya.</p>
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

        if st.button("📄 Generate & Unduh Laporan PDF", use_container_width=True):
            with st.spinner("Membuat laporan PDF..."):
                pdf_bytes = generate_pdf(m)
            st.download_button(
                label="⬇️ Unduh Laporan PDF",
                data=pdf_bytes,
                file_name="laporan_evaluasi_model_rf.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("Laporan PDF berhasil dibuat!")

    except FileNotFoundError:
        st.error("Hasil evaluasi tidak ditemukan. Jalankan `python scripts/03_evaluate.py` terlebih dahulu.")
