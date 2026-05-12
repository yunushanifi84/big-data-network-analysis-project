"""Keşifsel Veri Analizi (EDA) sayfası."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, PALETTE  # noqa: E402
from data_loader import load_gold_sample, ATTACK_COLORS  # noqa: E402

apply_theme()

st.markdown("# 🔍 Keşifsel Veri Analizi")
st.markdown(
    '<p style="color:#94A3B8">Gold tablosundan örneklenen veri üzerinde dağılım, '
    "zaman serisi ve eksik değer analizleri.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

df = load_gold_sample(limit=80_000)
if df.empty:
    st.warning("Gold tablosu boş — önce streaming pipeline'ını çalıştırın.")
    st.stop()

# ── Üst KPI'lar ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Örneklenen satır", f"{len(df):,}")
with k2:
    st.metric("Toplam kolon", f"{df.shape[1]}")
with k3:
    if "Attack_type" in df.columns:
        st.metric("Saldırı tipi sayısı", f"{df['Attack_type'].nunique()}")
    else:
        st.metric("Saldırı tipi", "—")
with k4:
    missing = df.isnull().sum().sum()
    pct = missing / (df.shape[0] * df.shape[1]) * 100
    st.metric("Eksik değer", f"{missing:,}", f"{pct:.2f}%")

# ── Saldırı tipi dağılımı ────────────────────────────────────────────────────
section("🎯 Saldırı Tipi Dağılımı", "Sınıf dengesizliği analizi")

if "Attack_type" in df.columns:
    vc = df["Attack_type"].value_counts().reset_index()
    vc.columns = ["Attack_type", "count"]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(
            vc,
            x="count",
            y="Attack_type",
            orientation="h",
            color="Attack_type",
            color_discrete_sequence=ATTACK_COLORS,
            labels={"count": "Satır sayısı", "Attack_type": ""},
        )
        fig.update_layout(showlegend=False, height=480, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            vc.head(8),
            values="count",
            names="Attack_type",
            color_discrete_sequence=ATTACK_COLORS,
            hole=0.55,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Zaman serisi ─────────────────────────────────────────────────────────────
section("📈 Zaman Serisi Trendi", "Saatlik mesaj hacmi")

ts_col = None
for c in ("ingestion_time", "timestamp", "frame_time"):
    if c in df.columns:
        ts_col = c
        break

if ts_col is not None:
    try:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        valid = ts.notna()
        if valid.sum() > 100:
            tdf = pd.DataFrame({
                "ts": ts[valid],
                "Attack_type": df.loc[valid, "Attack_type"] if "Attack_type" in df.columns else "all",
            })
            tdf["hour"] = tdf["ts"].dt.floor("h")
            hourly = tdf.groupby("hour").size().reset_index(name="count")
            fig = px.area(
                hourly, x="hour", y="count",
                labels={"hour": "Saat", "count": "Mesaj"},
            )
            fig.update_traces(line_color=PALETTE["primary"], fillcolor="rgba(99,102,241,0.25)")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{ts_col} sütununda geçerli tarih bulunamadı (parse oranı düşük).")
    except Exception as e:
        st.info(f"Zaman serisi oluşturulamadı: {e}")
else:
    st.info("Zaman bilgisi içeren bir kolon bulunamadı.")

# ── Eksik değer analizi ──────────────────────────────────────────────────────
section("🕳️ Eksik Değer Analizi", "En çok eksik içeren ilk 20 kolon")
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False).head(20)
if not miss.empty:
    mdf = pd.DataFrame({
        "column": miss.index,
        "missing_pct": (miss.values / len(df) * 100).round(2),
    })
    fig = px.bar(
        mdf, x="missing_pct", y="column", orientation="h",
        color="missing_pct",
        color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
        labels={"missing_pct": "% Eksik", "column": ""},
    )
    fig.update_layout(height=480, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("Eksik değer yok — Gold katmanı temiz ✨")

# ── Sayısal dağılımlar ──────────────────────────────────────────────────────
section("📊 Sayısal Feature Dağılımları", "Türetilmiş 5 özelliğin yoğunluk dağılımı")
ENG = ["traffic_asymmetry_ratio", "pkt_size_cv", "flow_intensity", "iat_regularity", "conn_efficiency"]
ENG = [c for c in ENG if c in df.columns]

if ENG:
    cols = st.columns(min(len(ENG), 3))
    for i, col in enumerate(ENG[:3]):
        with cols[i]:
            data = df[col].dropna()
            # Aşırı uçları kes (99. percentile)
            q99 = data.quantile(0.99) if not data.empty else 1
            data = data[(data >= data.quantile(0.01)) & (data <= q99)]
            fig = px.histogram(
                data, nbins=40, labels={"value": col},
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig.update_layout(height=260, showlegend=False, title=col)
            st.plotly_chart(fig, use_container_width=True)
    if len(ENG) > 3:
        cols2 = st.columns(min(len(ENG) - 3, 3))
        for i, col in enumerate(ENG[3:]):
            with cols2[i]:
                data = df[col].dropna()
                q99 = data.quantile(0.99) if not data.empty else 1
                data = data[(data >= data.quantile(0.01)) & (data <= q99)]
                fig = px.histogram(
                    data, nbins=40, labels={"value": col},
                    color_discrete_sequence=[PALETTE["secondary"]],
                )
                fig.update_layout(height=260, showlegend=False, title=col)
                st.plotly_chart(fig, use_container_width=True)

# ── Korelasyon ──────────────────────────────────────────────────────────────
section("🔥 Korelasyon Isı Haritası", "Türetilmiş özellikler + üst öneme sahip ham feature'lar")
focus_cols = ENG + [c for c in ("tcp_dstport", "tcp_srcport", "tcp_seq", "tcp_ack", "tcp_flags", "tcp_len") if c in df.columns]
focus_cols = [c for c in focus_cols if c in df.columns]
if len(focus_cols) >= 2:
    corr = df[focus_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, "#3B82F6"], [0.5, "#1E293B"], [1, "#EF4444"]],
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
