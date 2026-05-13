"""MLflow run kontrol scripti — eğitilmiş tüm modelleri raporlar."""
from mlflow.tracking import MlflowClient

c = MlflowClient("http://localhost:5000")
runs = c.search_runs(
    experiment_ids=["1"],
    max_results=100,
    order_by=["attributes.start_time DESC"],
)

print(f"\n{'='*120}")
print(f"  TOPLAM RUN SAYISI: {len(runs)}")
print(f"{'='*120}\n")

# Sütun başlıkları
print(f"{'Run Name':<42} {'Model':<24} {'Status':<10} {'AUC':>8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
print("-" * 120)

for r in runs:
    name = r.data.tags.get("mlflow.runName", "?")
    mtype = r.data.tags.get("model_type", "?")
    status = r.info.status
    m = r.data.metrics
    auc = m.get("auc_roc", m.get("cv_best_auc", None))
    acc = m.get("accuracy", None)
    f1 = m.get("f1_score", None)
    prec = m.get("precision", None)
    rec = m.get("recall", None)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    print(f"{name:<42} {mtype:<24} {status:<10} {fmt(auc):>8} {fmt(acc):>8} {fmt(f1):>8} {fmt(prec):>8} {fmt(rec):>8}")

print(f"\n{'='*120}")

# Model tipi başına en iyi run özeti
print("\n📊 Model tipi başına en iyi (AUC) run:")
print("-" * 120)
by_type = {}
for r in runs:
    if r.info.status != "FINISHED":
        continue
    mt = r.data.tags.get("model_type", "?")
    auc = r.data.metrics.get("auc_roc", r.data.metrics.get("cv_best_auc", 0))
    if mt not in by_type or auc > by_type[mt]["auc"]:
        by_type[mt] = {
            "auc": auc,
            "run_id": r.info.run_id,
            "name": r.data.tags.get("mlflow.runName", "?"),
            "acc": r.data.metrics.get("accuracy", 0),
            "f1": r.data.metrics.get("f1_score", 0),
        }

for mt, info in sorted(by_type.items(), key=lambda x: -x[1]["auc"]):
    print(f"  {mt:<28} AUC={info['auc']:.4f}  Acc={info['acc']:.4f}  F1={info['f1']:.4f}  ({info['name']})")
