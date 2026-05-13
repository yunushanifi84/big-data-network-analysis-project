"""
Big Data Project — Streamlit Dashboard
======================================
IoT ağ saldırılarının tespit edildiği Kafka → Spark → Delta → MLflow boru hattını
ve 5 modelli karşılaştırmayı sunan tek sayfalık modern kontrol paneli.

Çalıştırma:
    streamlit run streamlit_app/app.py

Docker:
    docker compose up streamlit-dashboard
"""
import streamlit as st

st.set_page_config(
    page_title="IoT Intrusion Detection — Big Data Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from theme import apply_theme  # noqa: E402

apply_theme()

# ── Üst Başlık ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown("# 🛡️ IoT Intrusion Detection")
    st.markdown(
        '<p style="color:#94A3B8;font-size:1.05rem;margin-top:-10px">'
        "Kafka → Spark Structured Streaming → Delta Lake → MLflow boru hattı · "
        "Edge-IIoTset üzerinde 5 modelli sınıflandırma karşılaştırması"
        "</p>",
        unsafe_allow_html=True,
    )
with col_right:
    st.markdown(
        '<div style="text-align:right;padding-top:10px">'
        '<span class="badge">Apache Kafka</span>'
        '<span class="badge">Spark 3.x</span>'
        '<span class="badge">Delta Lake</span>'
        '<span class="badge">MLflow</span>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Sidebar — proje özeti ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Big Data Project")
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.85rem">'
        "Streaming veri akışından makine öğrenmesi modellerine kadar uçtan uca "
        "büyük veri pipeline'ı."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**🧭 Sayfalar**")
    st.markdown(
        """
        - 📊 Genel Bakış
        - 🌊 Veri Akışı
        - 🔍 EDA
        - ⚙️ Feature Engineering
        - 🤖 Model Karşılaştırma
        - 🏆 En İyi Model
        """
    )
    st.markdown("---")
    if st.button("🔄 Yenile", key="home_refresh", use_container_width=True):
        get_layer_stats.clear()
        get_best_run_per_model.clear()
        load_mlflow_runs.clear()
        st.rerun()
    st.markdown(
        '<p style="color:#64748B;font-size:0.75rem">'
        "Soldaki menüden sayfa seçin · Veriler Delta Lake ve MLflow'dan canlı okunur"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Hoş geldin bloğu ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="info-card" style="border-color:rgba(139,92,246,0.4)">
        <h4>👋 Hoş geldin</h4>
        <p>
            Bu dashboard, <b>Edge-IIoTset</b> veri seti üzerinde gerçek zamanlı Kafka
            akışı simülasyonundan başlayarak <b>Bronze / Silver / Gold</b> Delta Lake
            katmanlarına, oradan 5 farklı sınıflandırma modeline uzanan tüm boru hattını
            tek bir görüntüde sergiler.
        </p>
        <p>
            Sol menüden bir sayfa seçerek detaylara in.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline diyagramı ───────────────────────────────────────────────────────
from theme import pipeline_diagram  # noqa: E402

pipeline_diagram([
    {"icon": "📡", "title": "Kafka Producer",
     "desc": "CSV → JSON streaming"},
    {"icon": "🪣", "title": "Bronze",
     "desc": "Ham JSON · Delta"},
    {"icon": "🧹", "title": "Silver",
     "desc": "Parse + temizlik"},
    {"icon": "✨", "title": "Gold",
     "desc": "ML-ready + 5 yeni feature"},
    {"icon": "🤖", "title": "ML Models",
     "desc": "5 sınıflandırıcı · CV"},
    {"icon": "📈", "title": "MLflow",
     "desc": "Tracking + artifact"},
])

# ── Hızlı KPI'lar ────────────────────────────────────────────────────────────
from data_loader import (  # noqa: E402
    get_layer_stats,
    get_best_run_per_model,
    load_mlflow_runs,
    ENGINEERED_FEATURES,
)

layer_df = get_layer_stats()
best_df = get_best_run_per_model()

k1, k2, k3, k4 = st.columns(4)
gold_row = layer_df[layer_df["layer"] == "Gold"].iloc[0] if not layer_df.empty else None

with k1:
    rows = gold_row["rows"] if gold_row is not None and gold_row["rows"] is not None else "—"
    rows_str = f"{rows:,}" if isinstance(rows, (int, float)) else str(rows)
    st.metric("Gold tablosu satır", rows_str)

with k2:
    cols = gold_row["columns"] if gold_row is not None and gold_row["columns"] is not None else "—"
    st.metric("Gold kolon sayısı", str(cols))

with k3:
    st.metric("Türetilmiş özellik", f"{len(ENGINEERED_FEATURES)}")

with k4:
    valid = best_df.dropna(subset=["accuracy"])
    if not valid.empty:
        top = valid.sort_values("accuracy", ascending=False).iloc[0]
        st.metric(
            "En iyi model",
            top["display"],
            f"acc = {top['accuracy']:.3f}",
        )
    else:
        st.metric("En iyi model", "—", "MLflow henüz boş")

st.markdown(
    '<p style="color:#64748B;font-size:0.85rem;margin-top:6px">'
    "ℹ️ Veriler <code>delta-storage/</code> ve <code>mlruns/mlflow.db</code> "
    "konumlarından her yenilemede canlı okunur (cache TTL: 60–300s)."
    "</p>",
    unsafe_allow_html=True,
)
