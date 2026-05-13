"""Model Karşılaştırma sayfası — 5 sınıflandırıcının metriklerle yan yana sunumu."""
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

st.markdown("# 🤖 Model Karşılaştırma")
st.markdown(
    '<p style="color:#94A3B8">5 farklı sınıflandırıcının MLflow\'a loglanmış en iyi run\'ları üzerinden karşılaştırması.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

best = get_best_run_per_model()

# ── Eksik MLflow uyarısı ─────────────────────────────────────────────────────
missing = best[~best["has_mlflow"]]["display"].tolist()
if missing:
    st.info(
        f"📌 MLflow'da henüz **{', '.join(missing)}** için tag'li run bulunamadı. "
        "İlgili modeli yeniden eğitince bu kartlar otomatik dolacak. "
        "Aşağıdaki karşılaştırma sadece mevcut run'lar üzerinden yapılır."
    )

# ── Üst metrik kartları ──────────────────────────────────────────────────────
section("🏅 Model Performansı (Accuracy)", "Tek bakışta sıralama")
sorted_best = best.sort_values("accuracy", ascending=False, na_position="last")
cols = st.columns(len(sorted_best))
for i, (_, row) in enumerate(sorted_best.iterrows()):
    with cols[i]:
        acc = row["accuracy"]
        acc_str = f"{acc:.3f}" if pd.notna(acc) else "—"
        f1_str = f"F1 = {row['f1_score']:.3f}" if pd.notna(row.get("f1_score")) else ""
        medal = "🥇" if i == 0 and pd.notna(acc) else "🥈" if i == 1 and pd.notna(acc) else "🥉" if i == 2 and pd.notna(acc) else "•"
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center;border-color:{row['color']}55">
                <div style="font-size:1.6rem">{medal}</div>
                <div style="color:#F1F5F9;font-weight:600;margin-top:4px">{row['display']}</div>
                <div style="color:{row['color']};font-weight:700;font-size:1.6rem;margin-top:6px">{acc_str}</div>
                <div style="color:#94A3B8;font-size:0.78rem">{f1_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Grouped bar — 4 metrik ───────────────────────────────────────────────────
section("📊 Çoklu Metrik Karşılaştırması", "Accuracy · F1 · Precision · Recall")
metrics_to_show = ["accuracy", "f1_score", "precision", "recall"]
plot_df = best.melt(
    id_vars=["display", "color"],
    value_vars=[m for m in metrics_to_show if m in best.columns],
    var_name="metric", value_name="value",
).dropna(subset=["value"])

if not plot_df.empty:
    fig = px.bar(
        plot_df, x="metric", y="value", color="display",
        barmode="group",
        color_discrete_map={m["display"]: m["color"] for m in MODELS},
        labels={"metric": "Metrik", "value": "Skor", "display": "Model"},
    )
    fig.update_layout(height=440, legend_title_text="")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Henüz metrik bulunamadı.")

# ── Radar chart ──────────────────────────────────────────────────────────────
section("🎯 Radar — Model Profilleri", "Modellerin güçlü/zayıf yanlarının görsel imzası")
radar_metrics = ["accuracy", "f1_score", "precision", "recall"]
have = [m for m in radar_metrics if m in best.columns]
if have:
    fig = go.Figure()
    for _, row in best.iterrows():
        vals = [row[m] if pd.notna(row.get(m)) else 0 for m in have]
        if max(vals) == 0:
            continue
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=have + [have[0]],
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

# ── Tablo ────────────────────────────────────────────────────────────────────
section("📋 Detay Tablo", "Tüm metrikler ve run bilgileri")
display_df = best[["display", "accuracy", "f1_score", "precision", "recall",
                   "auc_roc", "log_loss", "duration_sec", "has_mlflow"]].copy()
display_df.columns = ["Model", "Accuracy", "F1", "Precision", "Recall",
                      "AUC-ROC", "Log Loss", "Süre (sn)", "MLflow"]
for c in ("Accuracy", "F1", "Precision", "Recall", "AUC-ROC", "Log Loss"):
    display_df[c] = display_df[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
display_df["Süre (sn)"] = display_df["Süre (sn)"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
display_df["MLflow"] = display_df["MLflow"].map({True: "✅", False: "❌"})
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── MLflow run history ───────────────────────────────────────────────────────
section("📜 MLflow Run Geçmişi", "Son 15 başarılı run")
all_runs = load_mlflow_runs()
if not all_runs.empty:
    cols_to_show = ["name", "model_type", "accuracy", "f1_score", "duration_sec", "start_dt"]
    cols_to_show = [c for c in cols_to_show if c in all_runs.columns]
    history = all_runs[cols_to_show].head(15).copy()
    history.columns = [c.replace("_", " ").title() for c in history.columns]
    st.dataframe(history, use_container_width=True, hide_index=True)
else:
    st.info("MLflow DB boş.")
