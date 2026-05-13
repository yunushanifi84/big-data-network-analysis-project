"""
Adım 6.5 — Model 4: Gradient Boosted Trees (Multi-class via OneVsRest)
========================================================================
Spark MLlib'in GBTClassifier'ı doğrudan multi-class desteklemez.
Bu yüzden OneVsRest wrapper ile her sınıf için bir binary GBT eğitilir
(N sınıf → N tane binary GBT). En yüksek skoru veren sınıf seçilir.

Bu yaklaşım N kat daha yavaştır; bu yüzden CV grid'i sade tutulmuştur.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/04_gbt.py
"""
import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

sys.path.insert(0, "/opt/bitnami/spark")

from spark.spark_session import get_spark
from ml.utils import (
    run_ml_pipeline_multiclass,
    evaluate_model_multiclass,
    compute_confusion_matrix_multiclass,
    log_to_mlflow,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GBT (OneVsRest) multi-class eğitimi")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--cv-parallelism", type=int, default=2)
    return parser.parse_args()


def aggregate_feature_importance_ovr(ovr_model, feature_cols, num_classes):
    """
    OneVsRest içindeki her binary GBT modelinin featureImportances'ını
    aritmetik ortalama ile birleştir. Sonuçta her feature için
    sınıflar arası ortalama önem skoru elde edilir.
    """
    sub_models = ovr_model.models  # her sınıf için bir binary GBT
    n_features = len(feature_cols)
    agg = [0.0] * n_features

    for sm in sub_models:
        importances = sm.featureImportances.toArray().tolist()
        for j in range(n_features):
            agg[j] += importances[j] if j < len(importances) else 0.0
    agg = [v / max(len(sub_models), 1) for v in agg]

    pairs = sorted(zip(feature_cols, agg), key=lambda x: x[1], reverse=True)
    return pairs[:10], pairs


def confusion_matrix_to_text(cm: dict) -> str:
    labels = cm["labels"]
    matrix = cm["matrix"]
    lines = ["true_label," + ",".join(labels)]
    for i, name in enumerate(labels):
        row = ",".join(str(v) for v in matrix[i])
        lines.append(f"{name},{row}")
    return "\n".join(lines) + "\n"


def save_feature_importance_chart(top_features, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [f[0] for f in reversed(top_features)]
        values = [f[1] for f in reversed(top_features)]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, values, color="#FF9800", edgecolor="#E65100", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean Feature Importance (OneVsRest GBT)", fontsize=11)
        ax.set_title("GBT — Top 10 Feature Importance (Multi-class)",
                     fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Chart kaydedildi: {output_path}")
    except Exception as e:
        print(f"   ⚠️ Chart kaydedilemedi: {e}")


def save_feature_importance_csv(all_features, output_path):
    try:
        lines = ["rank,feature_name,importance"]
        for rank, (fname, imp) in enumerate(all_features, start=1):
            lines.append(f"{rank},{fname},{imp:.8f}")
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"   ✅ CSV kaydedildi: {output_path}")
    except Exception as e:
        print(f"   ⚠️ CSV kaydedilemedi: {e}")


def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 64)
    print("  GBT (OneVsRest) — Attack_type sınıflandırma (multi-class)")
    print("=" * 64)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode}")
    print("  ⚠️ OneVsRest: her sınıf için bir binary GBT → N kat daha yavaş.")

    spark = get_spark("Model-4-GBT-OneVsRest-Multiclass")

    print("\n[1/7] Multi-class veri pipeline hazırlanıyor...")
    train_df, test_df, feature_cols, label_index_model = run_ml_pipeline_multiclass(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    label_names = list(label_index_model.labels)
    num_classes = len(label_names)
    print(f"   📌 Feature: {len(feature_cols)} | Sınıf: {num_classes}")
    print(f"   📌 OneVsRest → {num_classes} adet binary GBT eğitilecek.")

    print("\n[2/7] GBT + OneVsRest pipeline kuruluyor...")
    # NOT: GBT weightCol destekler ama OneVsRest wrapper içinde
    # weightCol parametresi base classifier'a iletilir.
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=30,
        maxDepth=5,
        stepSize=0.1,
        seed=42,
    )
    ovr = OneVsRest(
        classifier=gbt,
        labelCol="label",
        featuresCol="features",
        weightCol="classWeight",
        parallelism=1,  # binary GBT'ler sıralı eğitilsin (RAM dostu)
    )
    pipeline = Pipeline(stages=[ovr])

    print("\n[3/7] Cross Validation...")
    # Multi-class + OneVsRest çok pahalı olduğu için grid'i çok dar tutuyoruz.
    if args.fast:
        max_iter_values = [20]
        max_depth_values = [3, 5]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            max_iter_values = [20, 50]
            max_depth_values = [3, 5, 7]
            num_folds = 3
        else:
            max_iter_values = [20, 50]
            max_depth_values = [3, 5]
            num_folds = 2

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, max_iter_values)
        .addGrid(gbt.maxDepth, max_depth_values)
        .build()
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1",
        ),
        numFolds=num_folds,
        seed=42,
        parallelism=args.cv_parallelism,
    )

    total_cv_runs = len(param_grid) * num_folds * num_classes
    print(
        f"   📌 Grid: {len(param_grid)} | Fold: {num_folds} | "
        f"OneVsRest çarpanı: {num_classes} | Toplam binary GBT fit: {total_cv_runs}"
    )

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    print(f"   ⏱️ CV süresi: {time.perf_counter() - cv_start:.2f}s")

    best_ovr_model = cv_model.bestModel.stages[-1]
    # Hangi parametreler kazandı? — ilk binary GBT'den okuyabiliriz
    first_sub = best_ovr_model.models[0]
    best_max_iter = int(first_sub.getOrDefault("maxIter"))
    best_max_depth = int(first_sub.getOrDefault("maxDepth"))
    best_step_size = float(first_sub.getOrDefault("stepSize"))
    print(
        "   ✅ En iyi parametreler: "
        f"maxIter={best_max_iter}, maxDepth={best_max_depth}, stepSize={best_step_size}"
    )

    print("\n[4/7] Test seti değerlendirme...")
    eval_start = time.perf_counter()
    predictions = cv_model.bestModel.transform(test_df)
    metrics = evaluate_model_multiclass(predictions, num_classes=num_classes)
    confusion = compute_confusion_matrix_multiclass(predictions, label_names=label_names)
    print(f"   ⏱️ Test süresi: {time.perf_counter() - eval_start:.2f}s")

    print("\n[5/7] CV sonuçları...")
    avg_metrics = cv_model.avgMetrics
    cv_details = []
    for i, (params, score) in enumerate(zip(param_grid, avg_metrics)):
        ps = ", ".join(f"{p.name}={v}" for p, v in params.items())
        marker = " ← best" if i == max(range(len(avg_metrics)),
                                       key=lambda x: avg_metrics[x]) else ""
        print(f"   {i+1:<3} F1={score:.4f}  {ps}{marker}")
        cv_details.append({"index": i, "score": float(score), "params": ps})
    cv_best_f1 = float(max(avg_metrics)) if avg_metrics else 0.0

    print("\n[6/7] Feature importance (sınıflar arası ortalama)...")
    top_features, all_features = aggregate_feature_importance_ovr(
        best_ovr_model, feature_cols, num_classes
    )
    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")
    chart_path = "/opt/bitnami/spark/ml/gbt_feature_importance.png"
    csv_path = "/opt/bitnami/spark/ml/gbt_feature_importance.csv"
    save_feature_importance_chart(top_features, chart_path)
    save_feature_importance_csv(all_features, csv_path)

    print("\n[7/7] MLflow'a loglanıyor...")
    best_params = {
        "maxIter": best_max_iter,
        "maxDepth": best_max_depth,
        "stepSize": best_step_size,
        "numClasses": int(num_classes),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_binary_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "wrapper": "OneVsRest",
        "weightCol": "classWeight",
        "label_column": "Attack_type",
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    metrics["cv_best_f1"] = cv_best_f1

    confusion_metrics_for_mlflow = {}
    for i, name in enumerate(label_names):
        confusion_metrics_for_mlflow[f"row_total_class_{i}"] = confusion["row_totals"][i]
        confusion_metrics_for_mlflow[f"per_class_acc_{i}"] = confusion["per_class_acc"][i]

    run_name = "gbt_v1_fast" if args.fast else "gbt_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="GBTClassifier",
        params=best_params,
        metrics={**metrics, **confusion_metrics_for_mlflow},
        model=cv_model.bestModel,
        confusion_matrix=None,
        feature_importance=top_features,
        tags={
            "task": "step_6_5",
            "model_index": "4",
            "classification_type": "multiclass",
            "label_column": "Attack_type",
            "num_classes": str(num_classes),
            "ensemble_method": "boosting",
            "multiclass_wrapper": "OneVsRest",
            "production_candidate": "true",
        },
    )

    try:
        import mlflow
        import os
        with mlflow.start_run(run_id=run_id):
            cv_csv = ["combination,avg_f1,params"]
            for d in cv_details:
                cv_csv.append(f"{d['index']+1},{d['score']:.6f},{d['params']}")
            mlflow.log_text("\n".join(cv_csv) + "\n", "cv_results.csv")
            mlflow.log_text(confusion_matrix_to_text(confusion), "confusion_matrix.csv")
            mlflow.log_text(
                "\n".join(f"{i},{name}" for i, name in enumerate(label_names)),
                "label_index_mapping.csv",
            )
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, "feature_importance")
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, "charts")
    except Exception as e:
        print(f"   ⚠️ Artifact hatası: {e}")

    print("\n" + "=" * 64)
    print("  ✅ GBT OneVsRest (Multi-class) tamamlandı!")
    print("=" * 64)
    print(f"  Run ID: {run_id}")
    print(f"  Sınıf sayısı: {num_classes}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  CV en iyi F1: {cv_best_f1:.4f}")
    print(f"  Toplam süre: {time.perf_counter() - total_start:.2f}s")
    print("=" * 64)

    spark.stop()


if __name__ == "__main__":
    main()
