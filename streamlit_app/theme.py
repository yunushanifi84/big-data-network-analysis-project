"""
Streamlit Dashboard — Tema ve ortak görsel yardımcılar.
Modern, koyu, sade. Plotly figürleri için ortak template.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from data_loader import PALETTE


# ── Global CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Genel arka plan */
    .stApp {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    /* Sidebar'dan 'app' girişini gizle */
    [data-testid="stSidebarNav"] li:first-child {
        display: none;
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #E2E8F0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h1 {
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Metric kart stili */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        transition: all 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(139, 92, 246, 0.55);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    /* Bilgi kutuları */
    .info-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }
    .info-card h4 {
        margin: 0 0 8px 0;
        color: #F1F5F9;
        font-size: 1.05rem;
    }
    .info-card p {
        color: #CBD5E1;
        margin: 4px 0;
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.22);
        color: #C7D2FE;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    /* Tablo */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    /* Pipeline akışı */
    .pipeline-flow {
        display: flex;
        gap: 8px;
        align-items: stretch;
        justify-content: space-between;
        margin: 20px 0 30px 0;
        flex-wrap: wrap;
    }
    .pipeline-node {
        flex: 1;
        min-width: 130px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 14px 12px;
        text-align: center;
        position: relative;
    }
    .pipeline-node .pn-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .pipeline-node .pn-title {
        color: #F1F5F9; font-weight: 600; font-size: 0.95rem;
    }
    .pipeline-node .pn-desc {
        color: #94A3B8; font-size: 0.78rem; margin-top: 4px;
    }
    .pipeline-arrow {
        align-self: center;
        color: #6366F1;
        font-size: 1.4rem;
        flex: 0 0 16px;
    }
    /* Section header */
    .section-header {
        border-left: 4px solid #6366F1;
        padding-left: 14px;
        margin: 30px 0 16px 0;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.4rem;
    }
    .section-header p {
        margin: 4px 0 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }
</style>
"""


def apply_theme():
    """Sayfa konfigürasyonu + global CSS uygula."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Plotly Template ──────────────────────────────────────────────────────────
def _register_plotly_template():
    tpl = go.layout.Template()
    tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
    tpl.layout.plot_bgcolor = "rgba(30, 41, 59, 0.3)"
    tpl.layout.font = dict(color=PALETTE["text"], family="Inter, system-ui, sans-serif", size=13)
    tpl.layout.xaxis = dict(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.2)",
        linecolor="rgba(148,163,184,0.3)",
    )
    tpl.layout.yaxis = dict(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.2)",
        linecolor="rgba(148,163,184,0.3)",
    )
    tpl.layout.colorway = [
        "#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316",
    ]
    tpl.layout.margin = dict(l=40, r=20, t=50, b=40)
    pio.templates["bigdata_dark"] = tpl
    pio.templates.default = "bigdata_dark"


_register_plotly_template()


# ── Yardımcı bileşenler ──────────────────────────────────────────────────────
def section(title: str, subtitle: str = "") -> None:
    """Sol kenarı vurgulu başlık bloğu."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, badges: list[str] | None = None) -> None:
    badge_html = ""
    if badges:
        badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
        badge_html = f'<div style="margin-top:10px">{badge_html}</div>'
    st.markdown(
        f"""
        <div class="info-card">
            <h4>{title}</h4>
            <p>{body}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_diagram(nodes: list[dict]) -> None:
    """nodes: [{'icon': '🌐', 'title': '...', 'desc': '...'}]"""
    parts = []
    for i, n in enumerate(nodes):
        parts.append(
            f"""
            <div class="pipeline-node">
                <div class="pn-icon">{n['icon']}</div>
                <div class="pn-title">{n['title']}</div>
                <div class="pn-desc">{n['desc']}</div>
            </div>
            """
        )
        if i < len(nodes) - 1:
            parts.append('<div class="pipeline-arrow">➜</div>')
    st.markdown(f'<div class="pipeline-flow">{"".join(parts)}</div>', unsafe_allow_html=True)
