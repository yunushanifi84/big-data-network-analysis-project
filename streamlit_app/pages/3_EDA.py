"""Keşifsel Veri Analizi (EDA) sayfası."""
import streamlit as st
import pandas as pd
import numpy as np
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

df = load_gold_sample()
if df.empty:
    st.warning("Gold tablosu boş — önce streaming pipeline'ını çalıştırın.")
    st.stop()

# ── Üst KPI'lar ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Toplam satır", f"{len(df):,}")
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
section("📈 Zaman Serisi Trendi", "Kayıt sırasına göre saldırı dağılımı")

if "Attack_type" in df.columns:
    # Kayıt sırasını zaman proksi olarak kullan (veri tek seferde stream edildiği için
    # gerçek timestamp'ler birbirine çok yakın — saatlik gruplama anlamsız kalır)
    window = max(len(df) // 200, 1)
    tdf = df[["Attack_type"]].copy()
    tdf["batch"] = (tdf.index // window)
    batch_counts = tdf.groupby(["batch", "Attack_type"]).size().reset_index(name="count")
    fig = px.area(
        batch_counts, x="batch", y="count", color="Attack_type",
        color_discrete_sequence=ATTACK_COLORS,
        labels={"batch": "Kayıt grubu", "count": "Mesaj sayısı", "Attack_type": "Saldırı"},
    )
    fig.update_layout(height=360, xaxis_title="Kayıt sırası →", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Attack_type kolonu bulunamadı.")

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

def _clean_for_hist(series: pd.Series) -> pd.Series:
    """Inf/NaN temizle, IQR tabanlı outlier kırp."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    q1, q3 = s.quantile(0.01), s.quantile(0.99)
    iqr = q3 - q1
    if iqr == 0:
        iqr = max(abs(q3), 1)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s[(s >= lower) & (s <= upper)]

if ENG:
    colors = [PALETTE["primary"]] * 3 + [PALETTE["secondary"]] * 2
    for row_start in range(0, len(ENG), 3):
        row_feats = ENG[row_start:row_start + 3]
        row_cols = st.columns(len(row_feats))
        for i, col in enumerate(row_feats):
            with row_cols[i]:
                data = _clean_for_hist(df[col])
                if data.empty:
                    st.info(f"{col}: tüm değerler Inf/NaN")
                    continue
                fig = px.histogram(
                    data, nbins=50, labels={"value": col},
                    color_discrete_sequence=[colors[row_start + i]],
                )
                fig.update_layout(height=260, showlegend=False, title=col)
                st.plotly_chart(fig, use_container_width=True)

# ── Korelasyon ──────────────────────────────────────────────────────────────
section("🔥 Korelasyon Isı Haritası", "Türetilmiş özellikler + üst öneme sahip ham feature'lar")
focus_cols = ENG + [c for c in ("tcp_dstport", "tcp_srcport", "tcp_seq", "tcp_ack", "tcp_flags", "tcp_len") if c in df.columns]
focus_cols = [c for c in focus_cols if c in df.columns]
if len(focus_cols) >= 2:
    corr_df = df[focus_cols].replace([np.inf, -np.inf], np.nan)
    corr = corr_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, "#3B82F6"], [0.5, "#1E293B"], [1, "#EF4444"]],
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
