"""Model Karşılaştırma sayfası — 5 multi-class sınıflandırıcının kıyaslaması."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Karşılaştırma", page_icon="🤖", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, PALETTE  # noqa: E402
from data_loader import (  # noqa: E402
    get_best_run_per_model,
    load_mlflow_runs,
    MODELS,
)

apply_theme()


# ── Metrik gösterim yardımcıları ─────────────────────────────────────────────
def _fmt_metric(v):
    """4 ondalık basıyoruz; 1'e çok yakın ama 1'den küçük değerler '1.0000' olarak
    yuvarlanıp 'mükemmel' algısı yaratmasın diye "0.9999" tabanına sıkıştırılır."""
    if not pd.notna(v):
        return "—"
    rounded = round(float(v), 4)
    if rounded >= 1.0 and float(v) < 1.0:
        return "0.9999"
    return f"{rounded:.4f}"


st.markdown("# 🤖 Model Karşılaştırma")
st.markdown(
    '<p style="color:#94A3B8">5 farklı <b>multi-class</b> sınıflandırıcının MLflow\'a loglanmış '
    'en iyi run\'ları üzerinden karşılaştırması '
    '(Attack_type — Normal + 14 saldırı tipi).</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Bağlam Uyarısı ───────────────────────────────────────────────────────────
st.info(
    "📌 **Multi-class değerlendirme**: Modeller artık saldırının **türünü** tahmin ediyor "
    "(DDoS_HTTP, Port_Scan, MITM, vb.). Bu binary 'attack var/yok' problemden çok daha zor. "
    "Accuracy sınıf dengesizliğinden etkilenir; **F1-Score (weighted)** daha güvenilir bir göstergedir. "
    "AUC-ROC multi-class için anlamsızdır, gösterilmez."
)

best = get_best_run_per_model()

# ── Eksik / yanlış sınıflandırma uyarısı ────────────────────────────────────
missing = best[~best["has_mlflow"]]["display"].tolist()
if missing:
    st.warning(
        f"⚠️ MLflow'da **{', '.join(missing)}** için run bulunamadı. "
        "İlgili modeli eğitince bu kart dolacak."
    )

# Eski binary run kullanan modeller (geçiş döneminde uyarı)
old_binary = best[
    best["classification_type"].fillna("").str.lower() == "binary"
]["display"].tolist()
if old_binary:
    st.warning(
        f"⚠️ **{', '.join(old_binary)}** modeli hâlâ eski binary run gösteriyor — "
        "yeni multi-class run'ı eğitilince otomatik güncellenecek."
    )

# ── Üst Metrik Kartları ──────────────────────────────────────────────────────
section(
    "🏅 Model Performansı (F1-Score sıralı)",
    "Multi-class için en güvenilir karşılaştırma metriği — weighted F1",
)
sorted_best = best.sort_values(
    by=["f1_score", "accuracy"], ascending=[False, False], na_position="last"
)
cols = st.columns(len(sorted_best))
for i, (_, row) in enumerate(sorted_best.iterrows()):
    with cols[i]:
        f1 = row.get("f1_score")
        acc = row["accuracy"]
        prec = row.get("precision")
        rec = row.get("recall")
        cls_type = row.get("classification_type") or "—"
        n_cls = row.get("num_classes") or "—"

        primary_str = _fmt_metric(f1)
        sub_str = (
            f"Acc = {_fmt_metric(acc)} · P = {_fmt_metric(prec)} · R = {_fmt_metric(rec)}"
        )
        medal = "🥇" if i == 0 and pd.notna(f1) else "🥈" if i == 1 and pd.notna(f1) else "🥉" if i == 2 and pd.notna(f1) else "•"
        type_tag = ""
        if isinstance(cls_type, str) and cls_type.lower() == "multiclass":
            type_tag = (
                f"<div style='color:#10B981;font-size:0.65rem;margin-top:4px'>"
                f"multi-class · {n_cls} sınıf</div>"
            )
        elif isinstance(cls_type, str) and cls_type.lower() == "binary":
            type_tag = (
                "<div style='color:#F59E0B;font-size:0.65rem;margin-top:4px'>"
                "⚠️ eski binary run</div>"
            )
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center;border-color:{row['color']}55">
                <div style="font-size:1.6rem">{medal}</div>
                <div style="color:#F1F5F9;font-weight:600;margin-top:4px">{row['display']}</div>
                <div style="color:{row['color']};font-weight:700;font-size:1.6rem;margin-top:6px">{primary_str}</div>
                <div style="color:#94A3B8;font-size:0.78rem">F1</div>
                <div style="color:#94A3B8;font-size:0.72rem;margin-top:4px">{sub_str}</div>
                {type_tag}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Grouped Bar — 4 Multi-class Metrik ───────────────────────────────────────
section(
    "📊 Çoklu Metrik Karşılaştırması",
    "Accuracy · F1 · Precision · Recall  (Weighted, multi-class)",
)
metrics_to_show = ["accuracy", "f1_score", "precision", "recall"]
metric_label = {
    "accuracy": "Accuracy",
    "f1_score": "F1",
    "precision": "Precision",
    "recall": "Recall",
}
plot_df = best.melt(
    id_vars=["display", "color"],
    value_vars=[m for m in metrics_to_show if m in best.columns],
    var_name="metric", value_name="value",
).dropna(subset=["value"])
plot_df["metric"] = plot_df["metric"].map(metric_label).fillna(plot_df["metric"])

if not plot_df.empty:
    fig = px.bar(
        plot_df, x="metric", y="value", color="display",
        barmode="group",
        category_orders={"metric": [metric_label[m] for m in metrics_to_show]},
        color_discrete_map={m["display"]: m["color"] for m in MODELS},
        labels={"metric": "Metrik", "value": "Skor", "display": "Model"},
    )
    fig.update_layout(height=460, legend_title_text="")
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Henüz metrik bulunamadı.")

# ── Radar Chart ──────────────────────────────────────────────────────────────
section("🎯 Radar — Model Profilleri", "Modellerin güçlü/zayıf yanları görsel olarak")
radar_metrics = ["accuracy", "f1_score", "precision", "recall"]
have = [m for m in radar_metrics if m in best.columns]
if have:
    radar_labels = [metric_label.get(m, m) for m in have]
    fig = go.Figure()
    for _, row in best.iterrows():
        vals = [row[m] if pd.notna(row.get(m)) else 0 for m in have]
        if max(vals) == 0:
            continue
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["display"],
            line=dict(color=row["color"], width=2),
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
            bgcolor="rgba(30,41,59,0.3)",
        ),
        height=480, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Detay Tablo ──────────────────────────────────────────────────────────────
section("📋 Detay Tablo", "Tüm metrikler ve run bilgileri")
display_df = best[[
    "display", "accuracy", "f1_score", "precision", "recall",
    "log_loss", "num_classes", "classification_type",
    "duration_sec", "has_mlflow",
]].copy()
display_df.columns = [
    "Model", "Accuracy", "F1", "Precision", "Recall",
    "Log Loss", "Sınıf #", "Tip", "Süre (sn)", "MLflow",
]
for c in ("Accuracy", "F1", "Precision", "Recall", "Log Loss"):
    display_df[c] = display_df[c].apply(_fmt_metric)
display_df["Süre (sn)"] = display_df["Süre (sn)"].apply(
    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
)
display_df["MLflow"] = display_df["MLflow"].map({True: "✅", False: "❌"})
display_df["Tip"] = display_df["Tip"].fillna("—")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── MLflow Run Geçmişi ───────────────────────────────────────────────────────
section("📜 MLflow Run Geçmişi", "Son 15 başarılı run")
all_runs = load_mlflow_runs()
if not all_runs.empty:
    cols_to_show = [
        "name", "model_type", "classification_type",
        "accuracy", "f1_score", "duration_sec", "start_dt",
    ]
    cols_to_show = [c for c in cols_to_show if c in all_runs.columns]
    history = all_runs[cols_to_show].head(15).copy()
    history.columns = [c.replace("_", " ").title() for c in history.columns]
    st.dataframe(history, use_container_width=True, hide_index=True)
else:
    st.info("MLflow DB boş.")
