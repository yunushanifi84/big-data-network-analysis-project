"""Veri Akışı sayfası — Bronze / Silver / Gold detayları + akış görselleştirmesi."""
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Veri Akışı", page_icon="🌊", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, pipeline_diagram, PALETTE  # noqa: E402
from data_loader import get_layer_stats  # noqa: E402

apply_theme()

st.markdown("# 🌊 Veri Akışı")
st.markdown(
    '<p style="color:#94A3B8">Kafka\'dan ML\'e: 3 katmanlı Delta Lake mimarisi (Medallion architecture).</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Pipeline ─────────────────────────────────────────────────────────────────
pipeline_diagram([
    {"icon": "📡", "title": "Producer", "desc": "kafka/producer.py"},
    {"icon": "🪣", "title": "Bronze", "desc": "Ham JSON"},
    {"icon": "🧹", "title": "Silver", "desc": "Şema + temizlik"},
    {"icon": "✨", "title": "Gold", "desc": "ML-ready"},
])

# ── Katman detayları ─────────────────────────────────────────────────────────
layer_df = get_layer_stats()

section("📦 Katman Detayları", "Her katmanın rolü ve mevcut durumu")

layer_meta = {
    "Bronze": {
        "icon": "🪣",
        "color": "#CD7F32",
        "purpose": "Kafka'dan gelen ham JSON mesajlarını şemasız olarak korur.",
        "writes": "writeStream → Delta · append-only",
        "schema": "json_payload (string) + Kafka metadata (topic, partition, offset, timestamp)",
        "tech": ["Spark Structured Streaming", "Delta Lake", "Append mode"],
    },
    "Silver": {
        "icon": "🧹",
        "color": "#C0C0C0",
        "purpose": "Şema uygulanır, null/duplike/format hataları temizlenir.",
        "writes": "Bronze → from_json + filtre → Delta",
        "schema": "Edge-IIoTset şeması (60+ kolon) + Attack_type label",
        "tech": ["from_json + schema inference", "Null/dup filtreleme", "Append mode"],
    },
    "Gold": {
        "icon": "✨",
        "color": "#FFD700",
        "purpose": "ML'e hazır feature tablosu — 5 yeni türetilmiş özellik dahil.",
        "writes": "Silver → FeatureEngineer → ml_ready_compact",
        "schema": "Numerik feature'lar + label_indexed (StringIndexer)",
        "tech": ["FeatureEngineer", "5 yeni feature", "StringIndexer", "VectorAssembler"],
    },
}

for _, row in layer_df.iterrows():
    meta = layer_meta.get(row["layer"], {})
    color = meta.get("color", "#6366F1")
    rows_v = f"{int(row['rows']):,}" if row["rows"] is not None else "—"
    cols_v = str(int(row["columns"])) if row["columns"] is not None else "—"

    st.markdown(
        f"""
        <div class="info-card" style="border-left:4px solid {color}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px">
                <div style="flex:2;min-width:280px">
                    <h4>{meta.get('icon','📦')} {row['layer']} Layer</h4>
                    <p>{meta.get('purpose','')}</p>
                    <p><b style="color:#94A3B8">Yazım:</b> <code>{meta.get('writes','')}</code></p>
                    <p><b style="color:#94A3B8">Şema:</b> {meta.get('schema','')}</p>
                    <div style="margin-top:8px">
                        {''.join(f'<span class="badge">{t}</span>' for t in meta.get('tech', []))}
                    </div>
                </div>
                <div style="flex:1;min-width:200px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Satır</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{rows_v}</div>
                    </div>
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Kolon</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{cols_v}</div>
                    </div>
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Boyut</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{row['size_mb']:.1f}<span style="font-size:0.7rem;color:#64748B"> MB</span></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sankey: veri akış hacmi ─────────────────────────────────────────────────
section("🔗 Veri Akış Hacmi", "Katmanlar arası satır akışı")

bronze = layer_df[layer_df["layer"] == "Bronze"]["rows"].iloc[0] if not layer_df.empty else None
silver = layer_df[layer_df["layer"] == "Silver"]["rows"].iloc[0] if not layer_df.empty else None
gold = layer_df[layer_df["layer"] == "Gold"]["rows"].iloc[0] if not layer_df.empty else None

if bronze and silver and gold:
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=24,
                thickness=22,
                line=dict(color="rgba(99,102,241,0.5)", width=1),
                label=[
                    f"Kafka<br>{bronze:,}",
                    f"Bronze<br>{bronze:,}",
                    f"Silver<br>{silver:,}",
                    f"Gold<br>{gold:,}",
                ],
                color=["#6366F1", "#CD7F32", "#C0C0C0", "#FFD700"],
            ),
            link=dict(
                source=[0, 1, 2],
                target=[1, 2, 3],
                value=[bronze, silver, gold],
                color=[
                    "rgba(99,102,241,0.35)",
                    "rgba(192,192,192,0.35)",
                    "rgba(255,215,0,0.35)",
                ],
            ),
        )
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Düşüş oranları
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Bronze → Silver", f"{silver:,}", f"{(silver-bronze)/bronze*100:+.1f}%")
    with c2:
        st.metric("Silver → Gold", f"{gold:,}", f"{(gold-silver)/silver*100:+.1f}%")
    with c3:
        st.metric("Toplam veri kaybı", f"{bronze-gold:,}", f"{(gold-bronze)/bronze*100:+.1f}%")
else:
    st.info("Katmanlar henüz veri ile dolmamış — pipeline'ı çalıştırın.")

# ── Komutlar ─────────────────────────────────────────────────────────────────
section("⚡ Çalıştırma Komutları", "Kendiniz deneyin")
st.code(
    """# 1) Tüm stack
docker compose up -d

# 2) Producer ile veri akışını başlat
docker compose up kafka-producer

# 3) Bronze → Silver → Gold streaming pipeline
docker compose exec spark-master spark-submit /opt/bitnami/spark/spark/run_streaming_pipeline.py

# 4) Model eğitimi
docker compose exec spark-master spark-submit /opt/bitnami/spark/ml/03_random_forest.py""",
    language="bash",
)
