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
section("📊 Sayısal Feature Dağılımları", "Saldırı tipine göre türetilmiş 5 özellik")
ENG = ["traffic_asymmetry_ratio", "pkt_size_cv", "flow_intensity", "iat_regularity", "conn_efficiency"]
ENG = [c for c in ENG if c in df.columns]

# Skew olan feature'larda log1p, tüm veri için aykırı kırpma
_USE_LOG = {"traffic_asymmetry_ratio", "pkt_size_cv", "flow_intensity"}

if ENG and "Attack_type" in df.columns:
    # Büyük veriyi küçült — violin plot için 30K yeterli
    sample = df[ENG + ["Attack_type"]].copy()
    if len(sample) > 30_000:
        sample = sample.sample(n=30_000, random_state=42)
    # Aşırı büyük / Inf temizle
    for col in ENG:
        sample[col] = pd.to_numeric(sample[col], errors="coerce")
        sample[col] = sample[col].replace([np.inf, -np.inf], np.nan)
        sample.loc[sample[col].abs() > 1e12, col] = np.nan
    # Çok nadir saldırı tiplerini çıkar (min 50 satır) — grafik netliği için
    counts = sample["Attack_type"].value_counts()
    valid_attacks = counts[counts >= 50].index.tolist()
    sample = sample[sample["Attack_type"].isin(valid_attacks)]
    for row_start in range(0, len(ENG), 3):
        row_feats = ENG[row_start:row_start + 3]
        row_cols = st.columns(len(row_feats))
        for i, col in enumerate(row_feats):
            with row_cols[i]:
                plot_df = sample[[col, "Attack_type"]].dropna()
                if plot_df.empty:
                    st.info(f"{col}: veri yok")
                    continue
                # Percentile kırp
                lo, hi = plot_df[col].quantile(0.02), plot_df[col].quantile(0.98)
                plot_df = plot_df[(plot_df[col] >= lo) & (plot_df[col] <= hi)]
                y_label = col
                if col in _USE_LOG:
                    plot_df[col] = np.log1p(plot_df[col].clip(lower=0))
                    y_label = f"log₁ₚ({col})"
                fig = px.box(
                    plot_df, x="Attack_type", y=col,
                    color="Attack_type",
                    color_discrete_sequence=ATTACK_COLORS,
                    labels={"Attack_type": "", col: y_label},
                )
                fig.update_layout(height=320, showlegend=False, title=col,
                                  xaxis_tickangle=-35, margin=dict(b=60))
                st.plotly_chart(fig, use_container_width=True)

# ── Korelasyon ──────────────────────────────────────────────────────────────
section("🔥 Korelasyon Isı Haritası", "Türetilmiş özellikler + üst öneme sahip ham feature'lar")
focus_cols = ENG + [c for c in ("tcp_dstport", "tcp_srcport", "tcp_seq", "tcp_ack", "tcp_flags", "tcp_len") if c in df.columns]
focus_cols = [c for c in focus_cols if c in df.columns]
if len(focus_cols) >= 2:
    corr_df = df[focus_cols].apply(pd.to_numeric, errors="coerce")
    corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
    corr_df[corr_df.abs() > 1e15] = np.nan
    corr = corr_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, "#3B82F6"], [0.5, "#1E293B"], [1, "#EF4444"]],
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
