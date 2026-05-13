"""En İyi Model sayfası — confusion matrix, per-class metrik, feature importance."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="En İyi Model", page_icon="🏆", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section, PALETTE  # noqa: E402
from data_loader import (  # noqa: E402
    get_best_run_per_model,
    get_run_metrics_full,
    load_feature_importance,
    MODELS,
    MODEL_BY_KEY,
)

apply_theme()

st.markdown("# 🏆 En İyi Model — Detay")
st.markdown(
    '<p style="color:#94A3B8">Confusion matrix, sınıf bazlı metrikler ve feature importance.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

best = get_best_run_per_model()

# ── Model seçici ─────────────────────────────────────────────────────────────
options = best["display"].tolist()
default_idx = 0
valid = best.dropna(subset=["accuracy"])
if not valid.empty:
    top = valid.sort_values("accuracy", ascending=False).iloc[0]["display"]
    if top in options:
        default_idx = options.index(top)

selected_display = st.selectbox("Model seç", options, index=default_idx)
row = best[best["display"] == selected_display].iloc[0]
model_key = row["model_key"]

# ── Üst KPI'lar ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
metric_pairs = [
    ("Accuracy", row.get("accuracy")),
    ("F1", row.get("f1_score")),
    ("Precision", row.get("precision")),
    ("Recall", row.get("recall")),
    ("AUC-ROC", row.get("auc_roc")),
]
for col, (name, v) in zip([m1, m2, m3, m4, m5], metric_pairs):
    with col:
        st.metric(name, f"{v:.4f}" if pd.notna(v) else "—")

# ── Confusion matrix ─────────────────────────────────────────────────────────
section("🔢 Confusion Matrix", "Sınıf bazında doğru/yanlış tahmin dağılımı")

full_metrics = get_run_metrics_full(row["run_id"]) if row.get("run_id") else {}

if full_metrics:
    # Binary CM (cm_TP/TN/FP/FN) → 2x2
    binary_keys = ("cm_TP", "cm_TN", "cm_FP", "cm_FN")
    if all(k in full_metrics for k in binary_keys):
        tp, tn, fp, fn = (full_metrics[k] for k in binary_keys)
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(
            cm,
            text_auto="d",
            labels=dict(x="Tahmin", y="Gerçek"),
            x=["Normal (0)", "Saldırı (1)"],
            y=["Normal (0)", "Saldırı (1)"],
            color_continuous_scale=[[0, "#1E293B"], [0.5, "#6366F1"], [1, "#EC4899"]],
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

    # Multiclass: per_class_acc_X, precision_class_X, recall_class_X
    class_indices = sorted({
        int(k.split("_")[-1])
        for k in full_metrics
        if k.startswith("per_class_acc_") and k.split("_")[-1].isdigit()
    })

    if class_indices:
        # Per-class precision / recall / f1 bar chart
        rows = []
        for ci in class_indices:
            rows.append({
                "class": f"C{ci}",
                "precision": full_metrics.get(f"precision_class_{ci}"),
                "recall": full_metrics.get(f"recall_class_{ci}"),
                "f1": full_metrics.get(f"f1_class_{ci}"),
                "acc": full_metrics.get(f"per_class_acc_{ci}"),
            })
        cdf = pd.DataFrame(rows)

        st.markdown("##### Sınıf Bazlı Metrikler")
        melt = cdf.melt(id_vars="class", value_vars=["precision", "recall", "f1"],
                        var_name="metric", value_name="value").dropna()
        fig = px.bar(
            melt, x="class", y="value", color="metric", barmode="group",
            labels={"class": "Sınıf", "value": "Skor", "metric": "Metrik"},
            color_discrete_map={
                "precision": "#6366F1",
                "recall": "#10B981",
                "f1": "#F59E0B",
            },
        )
        fig.update_layout(height=380)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Bu model için MLflow'da confusion matrix bilgisi bulunamadı.")

# ── ROC eğrisi (binary için) ─────────────────────────────────────────────────
section("📈 ROC Eğrisi", "True Positive Rate vs False Positive Rate")
if full_metrics and all(k in full_metrics for k in ("cm_TP", "cm_TN", "cm_FP", "cm_FN")):
    tp = full_metrics["cm_TP"]; tn = full_metrics["cm_TN"]
    fp = full_metrics["cm_FP"]; fn = full_metrics["cm_FN"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = row.get("auc_roc") or 0.5

    # Yaklaşık ROC (operating point + diagonal'a karşı düz çizgi)
    xs = [0, fpr, 1]
    ys = [0, tpr, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color=row["color"], width=3),
                             name=f"ROC (AUC ≈ {auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="rgba(148,163,184,0.4)", dash="dash"),
                             name="Random"))
    fig.update_layout(
        height=360,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "ℹ️ Operating point (model'in çalışma noktası) gösterilmiştir. "
        "Tam eğri için probabilistik skorlar gerekir."
    )
else:
    st.info("ROC için binary confusion matrix gerekli — bu run multiclass veya CM yok.")

# ── Feature Importance ───────────────────────────────────────────────────────
section("🌟 Feature Importance", "En etkili 15 özellik (horizontal bar)")
fi = load_feature_importance(model_key, top_n=15)
if not fi.empty:
    fig = px.bar(
        fi, x="importance", y="feature_name", orientation="h",
        color="importance",
        color_continuous_scale=[[0, MODEL_BY_KEY[model_key]["color"] + "33"],
                                 [1, MODEL_BY_KEY[model_key]["color"]]],
        labels={"importance": "Önem", "feature_name": ""},
    )
    fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Türetilmiş feature'ları vurgula
    ENG = {"traffic_asymmetry_ratio", "pkt_size_cv", "flow_intensity",
           "iat_regularity", "conn_efficiency"}
    eng_in_top = fi[fi["feature_name"].isin(ENG)]
    if not eng_in_top.empty:
        st.success(
            f"🎯 İlk 15 içinde **{len(eng_in_top)}** adet türetilmiş feature var: "
            + ", ".join(f"`{f}`" for f in eng_in_top["feature_name"])
        )
else:
    st.info(
        f"`{model_key}` için feature_importance CSV bulunamadı. "
        f"İlgili eğitim script'ini çalıştırın → `ml/{model_key}_feature_importance.csv` üretilir."
    )
