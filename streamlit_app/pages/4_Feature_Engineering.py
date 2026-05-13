"""Feature Engineering sayfası — 5 türetilmiş özellik için kart + dağılım + violin."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, PALETTE  # noqa: E402
from data_loader import load_gold_sample, ENGINEERED_FEATURES  # noqa: E402

apply_theme()

st.markdown("# ⚙️ Feature Engineering")
st.markdown(
    '<p style="color:#94A3B8">Edge-IIoTset üzerinde IoT saldırılarını tespit için '
    "üretilen 5 özel özellik — formül, iş mantığı ve dağılım.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Üst özet ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Türetilen feature", f"{len(ENGINEERED_FEATURES)}")
with c2:
    st.metric("Modül", "FeatureEngineer", "Spark DataFrame")
with c3:
    st.metric("Yazıldığı katman", "Gold", "delta-storage/gold/ml_ready_compact")

# ── Feature kartları ─────────────────────────────────────────────────────────
df = load_gold_sample(limit=60_000)

section("🧩 5 Türetilmiş Özellik", "Her biri belirli saldırı sınıflarını yakalamak için tasarlandı")

for f in ENGINEERED_FEATURES:
    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            badges = "".join(f'<span class="badge">{d}</span>' for d in f["detects"])
            st.markdown(
                f"""
                <div class="info-card" style="height:100%">
                    <div style="display:flex;align-items:center;gap:12px">
                        <div style="font-size:2.4rem">{f['icon']}</div>
                        <div>
                            <h4 style="margin:0">{f['title']}</h4>
                            <code style="color:#94A3B8;font-size:0.78rem">{f['name']}</code>
                        </div>
                    </div>
                    <div style="background:rgba(15,23,42,0.6);padding:10px 14px;border-radius:8px;
                                margin:14px 0;border-left:3px solid {PALETTE['primary']}">
                        <code style="color:#A5B4FC;font-size:0.9rem">{f['formula']}</code>
                    </div>
                    <p>{f['why']}</p>
                    <div style="margin-top:8px">
                        <div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:4px">Yakaladığı saldırılar</div>
                        {badges}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if not df.empty and f["name"] in df.columns:
                data = df[[f["name"], "Attack_type"]].dropna() if "Attack_type" in df.columns else df[[f["name"]]].dropna()
                # uçları kes
                col = data[f["name"]]
                q01, q99 = col.quantile(0.01), col.quantile(0.99)
                data = data[(col >= q01) & (col <= q99)]

                if "Attack_type" in data.columns:
                    # En kalabalık 6 saldırı tipi için violin
                    top_types = data["Attack_type"].value_counts().head(6).index.tolist()
                    sub = data[data["Attack_type"].isin(top_types)]
                    fig = px.violin(
                        sub, x="Attack_type", y=f["name"], color="Attack_type",
                        box=True, points=False,
                    )
                    fig.update_layout(height=320, showlegend=False, xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(data, x=f["name"], nbins=40)
                    fig.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"`{f['name']}` Gold tablosunda bulunamadı.")
    st.markdown("")

# ── Akış diyagramı ─────────────────────────────────────────────────────────
section("🔄 Üretim Akışı", "Silver kolonlarından Gold feature'larına dönüşüm")

st.markdown(
    """
    <div class="info-card">
    <p style="font-family:monospace;color:#CBD5E1;line-height:1.8;font-size:0.9rem">
    Silver DataFrame<br>
    &nbsp;&nbsp;↓ <code>FeatureEngineer(spark, silver_df)</code><br>
    &nbsp;&nbsp;↓ <code>.add_traffic_asymmetry_ratio()</code><br>
    &nbsp;&nbsp;↓ <code>.add_pkt_size_cv()</code><br>
    &nbsp;&nbsp;↓ <code>.add_flow_intensity()</code><br>
    &nbsp;&nbsp;↓ <code>.add_iat_regularity()</code><br>
    &nbsp;&nbsp;↓ <code>.add_conn_efficiency()</code><br>
    &nbsp;&nbsp;↓ <code>.create_all_features()</code><br>
    Gold DataFrame (5 yeni kolon eklendi)<br>
    &nbsp;&nbsp;↓<br>
    <code>delta-storage/gold/ml_ready_compact/</code>
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)
