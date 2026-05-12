"""Veri Akışı sayfası — Bronze / Silver / Gold detayları + akış görselleştirmesi."""
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Veri Akışı", page_icon="🌊", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime  # noqa: E402
from theme import apply_theme, section, pipeline_diagram, PALETTE  # noqa: E402
from data_loader import (  # noqa: E402
    get_layer_stats,
    get_kafka_topic_offsets,
    get_layer_freshness,
)
from streamlit_autorefresh import st_autorefresh  # noqa: E402

apply_theme()

# ── Sidebar: canlı izleme kontrolleri ────────────────────────────────────────────
auto_refresh = False
refresh_sec = 10
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ Canlı İzleme")
    auto_refresh = st.toggle("Otomatik Yenile", value=True, key="live_auto_refresh")
    if auto_refresh:
        refresh_sec = st.select_slider(
            "Yenileme aralığı",
            options=[1, 5, 10, 30, 60],
            value=1,
            format_func=lambda x: f"{x} saniye",
        )
    else:
        if st.button("🔄 Şimdi Yenile", key="manual_refresh_btn", use_container_width=True):
            get_kafka_topic_offsets.clear()
            get_layer_freshness.clear()
            get_layer_stats.clear()
            st.rerun()

if auto_refresh:
    _refresh_count = st_autorefresh(interval=refresh_sec * 1000, key="live_refresh_ctr")
    get_kafka_topic_offsets.clear()
    get_layer_freshness.clear()
    # Her 30 yenilemede layer_stats'ı da temizle → zaman serisi gerçek zamanlı güncellenir
    if _refresh_count % 30 == 0:
        get_layer_stats.clear()

st.markdown("# 🌊 Veri Akışı")
st.markdown(
    '<p style="color:#94A3B8">Kafka\'dan ML\'e: 3 katmanlı Delta Lake mimarisi (Medallion architecture).</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# layer_df erken fetch — hem Canlı İzleme hem Animasyon hem de Katman Detayları kullanır
layer_df = get_layer_stats()
_layer_rows: dict = {}
for _, _lr in layer_df.iterrows():
    try:
        _layer_rows[_lr["layer"]] = int(_lr["rows"])
    except (TypeError, ValueError):
        _layer_rows[_lr["layer"]] = 0

# ── Canlı İzleme ─────────────────────────────────────────────────────────────────────────────
section("🟥 Canlı İzleme", "Producer ve streaming pipeline'in anlık durumu")

c_kafka, c_layers = st.columns([1, 2])

with c_kafka:
    kstats = get_kafka_topic_offsets()
    k_status = kstats.get("status", "error")
    if k_status == "ok":
        k_color, k_dot, k_label = "#10B981", "🟢", "Bağlı / Aktif"
    elif k_status == "no_topic":
        k_color, k_dot, k_label = "#F59E0B", "🟡", "Topic bulunamadı"
    else:
        k_color, k_dot, k_label = "#EF4444", "🔴", "Broker'a ulaşılamıyor"

    err_html = (
        f'<div style="color:#EF4444;font-size:0.7rem;margin-top:6px;word-break:break-all">'
        f'{kstats.get("error", "")[:80]}</div>'
    ) if k_status == "error" else ""

    st.markdown(
        f"""
        <div class="info-card" style="border-left:4px solid {k_color}">
            <h4>📡 Kafka Producer</h4>
            <div style="font-size:2.2rem;margin:6px 0">{k_dot}</div>
            <div style="color:{k_color};font-weight:700;font-size:1rem;margin-bottom:14px">{k_label}</div>
            <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                <div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;margin-bottom:4px">Topic Mesaj (end offset)</div>
                <div style="color:#F1F5F9;font-weight:700;font-size:1.6rem">{kstats.get('total_messages', 0):,}</div>
            </div>
            <div style="color:#64748B;font-size:0.75rem;margin-top:10px;text-align:center">
                {kstats.get('topic', '—')} &nbsp;·&nbsp; {kstats.get('partitions', 0)} partition
            </div>
            {err_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_layers:
    freshness = get_layer_freshness()
    lmeta_live = {
        "Bronze": {"color": "#CD7F32", "icon": "🪣"},
        "Silver": {"color": "#C0C0C0", "icon": "🧹"},
        "Gold":   {"color": "#FFD700", "icon": "✨"},
    }
    rows_html = ""
    for lname, lm in lmeta_live.items():
        fi = freshness.get(lname, {})
        exists = fi.get("exists", False)
        active = fi.get("active", False)
        last_mod = fi.get("last_modified") or "—"
        age_sec = fi.get("age_sec")
        has_data = _layer_rows.get(lname, 0) > 0
        if not exists:
            dot, dot_color, status_lbl, age_str = "⚫", "#64748B", "Henüz oluşturulmadı", "—"
        elif active:
            dot, dot_color, status_lbl = "🟢", "#10B981", "Aktif — yazıyor"
            age_str = f"{age_sec}s önce" if age_sec is not None else "—"
        elif has_data:
            dot, dot_color, status_lbl = "🔵", "#3B82F6", "Tamamlandı"
            age_str = f"{age_sec}s önce" if age_sec is not None else "—"
        else:
            dot, dot_color, status_lbl = "🟡", "#F59E0B", "Bekliyor — veri yok"
            age_str = f"{age_sec}s önce" if age_sec is not None else "—"
        rows_html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:rgba(15,23,42,0.45);border-radius:10px;margin-bottom:8px;
                    border-left:3px solid {lm['color']}">
            <div style="font-size:1.4rem">{dot}</div>
            <div style="flex:1">
                <div style="color:#F1F5F9;font-weight:600">{lm['icon']} {lname}</div>
                <div style="color:{dot_color};font-size:0.78rem;margin-top:2px">{status_lbl}</div>
            </div>
            <div style="text-align:right">
                <div style="color:#94A3B8;font-size:0.7rem">Son yazım</div>
                <div style="color:#E2E8F0;font-size:0.85rem;font-weight:600">{last_mod}</div>
                <div style="color:#64748B;font-size:0.7rem">{age_str}</div>
            </div>
        </div>
        """

    _now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="info-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <h4 style="margin:0">🔄 Streaming Katman Durumu</h4>
                <span style="color:#64748B;font-size:0.75rem">Güncellendi: {_now_str}</span>
            </div>
            <p style="color:#64748B;font-size:0.78rem;margin-bottom:10px">
                Son parquet/delta-log değişikliğinden ≤120s geçmişse aktif sayılır
            </p>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Pipeline ─────────────────────────────────────────────────────────────────────────────
pipeline_diagram([
    {"icon": "📡", "title": "Producer", "desc": "kafka/producer.py"},
    {"icon": "🪣", "title": "Bronze", "desc": "Ham JSON"},
    {"icon": "🧹", "title": "Silver", "desc": "Şema + temizlik"},
    {"icon": "✨", "title": "Gold", "desc": "ML-ready"},
])

# ── Canlı Akış Animasyonu ─────────────────────────────────────────────────────
st.markdown("---")
section("🎬 Canlı Akış Animasyonu", "Kafka → Bronze → Silver → Gold gerçek zamanlı veri hareketi")

_b_active = freshness.get("Bronze", {}).get("active", False)
_s_active = freshness.get("Silver", {}).get("active", False)
_g_active = freshness.get("Gold", {}).get("active", False)
_k_ok     = kstats.get("status") == "ok"

# Tüm değerleri önce düz Python değişkenlerine at — f-string içinde iç içe
# quote ve koşul olmadığı için her Python sürümünde güvenle çalışır.

def _make_dots(color, active, count=3):
    if not active:
        return ""
    step = 1.2 / count
    parts = []
    for i in range(count):
        parts.append(
            '<div class="anim-dot" style="background:' + color + ';'
            'animation-duration:1.2s;animation-delay:' + f"{i * step:.2f}" + 's"></div>'
        )
    return "".join(parts)

# Kafka
_k_op    = "1"      if _k_ok     else "0.4"
_k_sc    = "#10B981" if _k_ok    else "#EF4444"
_k_st    = "&#9679; Aktif"       if _k_ok     else "&#9679; Kapalı"
_k_anim  = "animation:glow-pulse 1.4s ease-in-out infinite;--glow:rgba(99,102,241,0.65);" if _k_ok else ""
_k_pbg   = "rgba(205,127,50,0.45)"  if _k_ok     else "rgba(205,127,50,0.12)"
_k_dots  = _make_dots("#E8964A", _k_ok)

# Bronze
_b_done  = (not _b_active) and _layer_rows.get("Bronze", 0) > 0
_b_op    = "1"       if (_b_active or _b_done) else "0.4"
_b_sc    = "#10B981" if _b_active else ("#3B82F6" if _b_done else "#64748B")
_b_st    = "&#9679; Yazıyor" if _b_active else ("&#10003; Tamamland&#305;" if _b_done else "&#9675; Bekliyor")
_b_anim  = "animation:glow-pulse 1.4s ease-in-out infinite;--glow:rgba(205,127,50,0.65);" if _b_active else ""
_b_pbg   = "rgba(192,192,192,0.45)" if _b_active else "rgba(192,192,192,0.12)"
_b_dots  = _make_dots("#C0C0C0", _b_active)

# Silver
_s_done  = (not _s_active) and _layer_rows.get("Silver", 0) > 0
_s_op    = "1"       if (_s_active or _s_done) else "0.4"
_s_sc    = "#10B981" if _s_active else ("#3B82F6" if _s_done else "#64748B")
_s_st    = "&#9679; Yazıyor" if _s_active else ("&#10003; Tamamland&#305;" if _s_done else "&#9675; Bekliyor")
_s_anim  = "animation:glow-pulse 1.4s ease-in-out infinite;--glow:rgba(192,192,192,0.65);" if _s_active else ""
_s_pbg   = "rgba(255,215,0,0.45)"   if _s_active else "rgba(255,215,0,0.12)"
_s_dots  = _make_dots("#FFD700", _s_active)

# Gold
_g_done  = (not _g_active) and _layer_rows.get("Gold", 0) > 0
_g_op    = "1"       if (_g_active or _g_done) else "0.4"
_g_sc    = "#10B981" if _g_active else ("#3B82F6" if _g_done else "#64748B")
_g_st    = "&#9679; Yazıyor" if _g_active else ("&#10003; Tamamland&#305;" if _g_done else "&#9675; Bekliyor")
_g_anim  = "animation:glow-pulse 1.4s ease-in-out infinite;--glow:rgba(255,215,0,0.65);"  if _g_active else ""

_anim_html = (
    "<style>"
    "@keyframes particle-flow{"
    "0%{left:-10px;opacity:0}"
    "15%{opacity:1}"
    "85%{opacity:1}"
    "100%{left:calc(100% + 10px);opacity:0}"
    "}"
    "@keyframes glow-pulse{"
    "0%,100%{box-shadow:0 0 4px 2px var(--glow,rgba(255,255,255,0.2))}"
    "50%{box-shadow:0 0 18px 7px var(--glow,rgba(255,255,255,0.2))}"
    "}"
    ".af-wrap{display:flex;align-items:center;padding:28px 16px;"
    "background:rgba(15,23,42,0.55);border-radius:16px;"
    "border:1px solid rgba(99,102,241,0.18)}"
    ".af-box{display:flex;flex-direction:column;align-items:center;"
    "justify-content:center;width:118px;height:100px;border-radius:14px;"
    "font-weight:700;font-size:0.82rem;flex-shrink:0;position:relative;z-index:2}"
    ".af-pipe{position:relative;flex:1;height:6px;border-radius:3px;"
    "overflow:visible;margin:0 -1px}"
    ".af-dot{position:absolute;top:50%;transform:translateY(-50%);"
    "width:10px;height:10px;border-radius:50%;"
    "animation:particle-flow linear infinite}"
    "</style>"
    # ── Kafka ──
    '<div class="af-wrap">'
    '<div class="af-box" style="background:rgba(99,102,241,0.12);'
    "border:2px solid #6366F1;opacity:" + _k_op + ";color:#A5B4FC;" + _k_anim + '">'
    '<div style="font-size:1.8rem">&#128225;</div>'
    '<div style="margin-top:4px">Kafka</div>'
    '<div style="font-size:0.65rem;color:' + _k_sc + ';margin-top:3px">' + _k_st + '</div>'
    "</div>"
    # ── Kafka→Bronze pipe ──
    '<div class="af-pipe" style="background:' + _k_pbg + '">' + _k_dots + "</div>"
    # ── Bronze ──
    '<div class="af-box" style="background:rgba(205,127,50,0.12);'
    "border:2px solid #CD7F32;opacity:" + _b_op + ";color:#D4904E;" + _b_anim + '">'
    '<div style="font-size:1.8rem">&#129379;</div>'
    '<div style="margin-top:4px">Bronze</div>'
    '<div style="font-size:0.65rem;color:' + _b_sc + ';margin-top:3px">' + _b_st + '</div>'
    "</div>"
    # ── Bronze→Silver pipe ──
    '<div class="af-pipe" style="background:' + _b_pbg + '">' + _b_dots + "</div>"
    # ── Silver ──
    '<div class="af-box" style="background:rgba(192,192,192,0.08);'
    "border:2px solid #C0C0C0;opacity:" + _s_op + ";color:#C0C0C0;" + _s_anim + '">'
    '<div style="font-size:1.8rem">&#129529;</div>'
    '<div style="margin-top:4px">Silver</div>'
    '<div style="font-size:0.65rem;color:' + _s_sc + ';margin-top:3px">' + _s_st + '</div>'
    "</div>"
    # ── Silver→Gold pipe ──
    '<div class="af-pipe" style="background:' + _s_pbg + '">' + _s_dots + "</div>"
    # ── Gold ──
    '<div class="af-box" style="background:rgba(255,215,0,0.08);'
    "border:2px solid #FFD700;opacity:" + _g_op + ";color:#FFD700;" + _g_anim + '">'
    '<div style="font-size:1.8rem">&#10024;</div>'
    '<div style="margin-top:4px">Gold</div>'
    '<div style="font-size:0.65rem;color:' + _g_sc + ';margin-top:3px">' + _g_st + '</div>'
    "</div>"
    "</div>"
)

st.markdown(_anim_html, unsafe_allow_html=True)

# ── Canlı Zaman Serisi ────────────────────────────────────────────────────────
_MAX_HIST = 120

if "flow_history" not in st.session_state:
    st.session_state.flow_history = []

_b = layer_df[layer_df["layer"] == "Bronze"]["rows"].values[0] if not layer_df.empty else None
_s = layer_df[layer_df["layer"] == "Silver"]["rows"].values[0] if not layer_df.empty else None
_g = layer_df[layer_df["layer"] == "Gold"]["rows"].values[0] if not layer_df.empty else None

if any(v is not None for v in [_b, _s, _g]):
    st.session_state.flow_history.append(
        {"ts": datetime.now(), "bronze": _b, "silver": _s, "gold": _g}
    )
    if len(st.session_state.flow_history) > _MAX_HIST:
        st.session_state.flow_history = st.session_state.flow_history[-_MAX_HIST:]

if len(st.session_state.flow_history) >= 1:
    _hist  = st.session_state.flow_history
    _times = [h["ts"] for h in _hist]
    _ts_fig = go.Figure()
    for _lname, _color, _fill in [
        ("bronze", "#CD7F32", "rgba(205,127,50,0.10)"),
        ("silver", "#C0C0C0", "rgba(192,192,192,0.10)"),
        ("gold",   "#FFD700", "rgba(255,215,0,0.10)"),
    ]:
        _ts_fig.add_trace(go.Scatter(
            x=_times,
            y=[h.get(_lname) for h in _hist],
            mode="lines",
            name=_lname.capitalize(),
            line=dict(color=_color, width=2),
            fill="tozeroy",
            fillcolor=_fill,
        ))
    _ts_fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#94A3B8")),
        xaxis=dict(showgrid=False, color="#64748B", tickfont=dict(color="#64748B")),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(100,116,139,0.15)",
            color="#64748B",
            tickfont=dict(color="#64748B"),
            title=dict(text="Satır sayısı", font=dict(color="#64748B")),
        ),
    )
    st.plotly_chart(_ts_fig, use_container_width=True)
else:
    st.caption("Zaman serisi için en az 2 veri noktası bekleniyor…")

st.markdown("---")

# ── Katman detayları ─────────────────────────────────────────────────────────
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
