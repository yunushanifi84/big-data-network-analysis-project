"""Genel Bakış sayfası — proje hikâyesi + mimari + üst düzey metrikler."""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Genel Bakış", page_icon="📊", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, info_card, pipeline_diagram
from data_loader import (
    get_layer_stats,
    get_best_run_per_model,
    load_mlflow_runs,
    ENGINEERED_FEATURES,
    MODELS,
)

apply_theme()

with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Yenile", key="genel_refresh", use_container_width=True):
        get_layer_stats.clear()
        get_best_run_per_model.clear()
        load_mlflow_runs.clear()
        st.rerun()

st.markdown("# 📊 Genel Bakış")
st.markdown(
    '<p style="color:#94A3B8">Edge-IIoTset üzerinde uçtan uca Big Data + ML boru hattının yüksek-düzey özeti.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Proje hikâyesi ───────────────────────────────────────────────────────────
section("🎯 Problem", "Ne çözüyoruz?")
st.markdown(
    """
    <div class="info-card">
    <p>
    IoT cihazlarının ürettiği ağ trafiğinde <b>15 farklı saldırı tipi</b> (DDoS, Port Scan,
    SQL Injection, MITM vb.) ile normal trafik birbirine karışmış durumdadır. Bu projede:
    </p>
    <ul style="color:#CBD5E1;line-height:1.7">
    <li>Trafik <b>Kafka</b> üzerinden gerçek-zamanlı simüle edilir,</li>
    <li><b>Spark Structured Streaming</b> ile okunur ve <b>Delta Lake</b>'in 3 katmanına yazılır,</li>
    <li>5 yeni <b>türetilmiş özellik</b> üretilir,</li>
    <li>5 farklı sınıflandırıcı eğitilip <b>MLflow</b> ile karşılaştırılır.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Mimari ───────────────────────────────────────────────────────────────────
section("🏗️ Mimari", "Veri yolculuğu")
pipeline_diagram([
    {"icon": "📡", "title": "Kafka Producer",
     "desc": "CSV → JSON · ayarlanabilir hız"},
    {"icon": "🪣", "title": "Bronze",
     "desc": "Ham JSON · şemasız"},
    {"icon": "🧹", "title": "Silver",
     "desc": "Şema · null/dup temizlik"},
    {"icon": "✨", "title": "Gold",
     "desc": "ML-ready · 5 yeni feature"},
    {"icon": "🤖", "title": "ML Eğitim",
     "desc": "5 model · CV · MLflow"},
    {"icon": "📈", "title": "Dashboard",
     "desc": "Streamlit · Plotly"},
])

# ── Katman istatistikleri ────────────────────────────────────────────────────
section("📦 Delta Lake Katmanları", "Her katmanın anlık durumu")
layer_df = get_layer_stats()
if not layer_df.empty:
    cols = st.columns(3)
    icons = {"Bronze": "🪣", "Silver": "🧹", "Gold": "✨"}
    for idx, (_, row) in enumerate(layer_df.iterrows()):
        with cols[idx]:
            try:
                rows_v = f"{int(row['rows']):,}"
            except (ValueError, TypeError):
                rows_v = "—"
            try:
                cols_v = str(int(row["columns"]))
            except (ValueError, TypeError):
                cols_v = "—"
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>{icons.get(row['layer'], '📦')} {row['layer']}</h4>
                    <p style="color:#94A3B8;font-size:0.85rem;margin:0">{row['description']}</p>
                    <div style="margin-top:14px;display:flex;justify-content:space-between">
                        <div>
                            <div style="color:#64748B;font-size:0.75rem;text-transform:uppercase">Satır</div>
                            <div style="color:#F1F5F9;font-size:1.4rem;font-weight:700">{rows_v}</div>
                        </div>
                        <div>
                            <div style="color:#64748B;font-size:0.75rem;text-transform:uppercase">Kolon</div>
                            <div style="color:#F1F5F9;font-size:1.4rem;font-weight:700">{cols_v}</div>
                        </div>
                        <div>
                            <div style="color:#64748B;font-size:0.75rem;text-transform:uppercase">Boyut</div>
                            <div style="color:#F1F5F9;font-size:1.4rem;font-weight:700">{row['size_mb']:.1f} <span style="font-size:0.75rem;color:#64748B">MB</span></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Modeller özeti ───────────────────────────────────────────────────────────
section("🤖 Modeller", "Sınıflandırma yarışmacıları")
model_cols = st.columns(len(MODELS))
best_df = get_best_run_per_model()
for i, m in enumerate(MODELS):
    with model_cols[i]:
        row = best_df[best_df["model_key"] == m["key"]]
        acc = row["accuracy"].iloc[0] if not row.empty else None
        acc_str = f"{acc:.3f}" if acc is not None else "—"
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center;border-color:{m['color']}55">
                <div style="width:46px;height:46px;border-radius:50%;background:{m['color']}22;
                            border:2px solid {m['color']};margin:0 auto 8px auto;
                            display:flex;align-items:center;justify-content:center;
                            color:{m['color']};font-weight:700;font-size:1.1rem">
                    {m['display'][0]}
                </div>
                <div style="color:#F1F5F9;font-weight:600;font-size:0.92rem">{m['display']}</div>
                <div style="color:#94A3B8;font-size:0.78rem;margin-top:6px">Accuracy</div>
                <div style="color:{m['color']};font-weight:700;font-size:1.35rem">{acc_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Türetilmiş feature'lar ──────────────────────────────────────────────────
section("⚙️ Türetilmiş 5 Özellik", "Edge-IIoTset için özel mühendislik")
fcols = st.columns(5)
for i, f in enumerate(ENGINEERED_FEATURES):
    with fcols[i]:
        st.markdown(
            f"""
            <div class="info-card" style="min-height:160px">
                <div style="font-size:1.8rem">{f['icon']}</div>
                <div style="color:#F1F5F9;font-weight:600;margin-top:6px">{f['title']}</div>
                <div style="color:#64748B;font-family:monospace;font-size:0.74rem;margin-top:6px">{f['name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<p style="color:#64748B;font-size:0.85rem;text-align:center">'
    "Daha fazla detay için soldaki menüden ilgili sayfayı seçin →"
    "</p>",
    unsafe_allow_html=True,
)
