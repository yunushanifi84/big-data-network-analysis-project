"""
Adım 6.3 — Model 2: Decision Tree (Multi-class)
==================================================
Saldırı tipi (Attack_type) için yorumlanabilir bir multi-class Decision Tree.
Cross validation ile hiperparametre seçer, MLflow'a loglar.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/02_decision_tree.py
"""
import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
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
    parser = argparse.ArgumentParser(description="Decision Tree multi-class eğitimi")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--cv-parallelism", type=int, default=2)
    return parser.parse_args()


def extract_feature_importance(cv_model, feature_cols):
    dt_model = cv_model.bestModel.stages[-1]
    importances = dt_model.featureImportances.toArray().tolist()
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return pairs[:10]


def analyze_tree_structure(cv_model):
    dt_model = cv_model.bestModel.stages[-1]
    depth = dt_model.depth
    num_nodes = dt_model.numNodes
    debug_string = dt_model.toDebugString
    debug_lines = debug_string.split("\n")
    debug_preview = "\n".join(debug_lines[:50])
    if len(debug_lines) > 50:
        debug_preview += f"\n... ({len(debug_lines) - 50} satır daha)"

    print("\n🌳 Karar Ağacı Yapı:")
    print(f"   Derinlik: {depth} | Düğüm: {num_nodes} | Yaprak: ~{(num_nodes+1)//2}")
    return {
        "depth": depth,
        "num_nodes": num_nodes,
        "num_leaves": (num_nodes + 1) // 2,
        "debug_string_preview": debug_preview,
    }


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
        bars = ax.barh(names, values, color="#2196F3", edgecolor="#1565C0", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Feature Importance (Gini/Entropy)", fontsize=11)
        ax.set_title("Decision Tree — Top 10 Feature Importance (Multi-class)",
                     fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Chart kaydedildi: {output_path}")
    except Exception as e:
        print(f"   ⚠️ Chart kaydedilemedi: {e}")


def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 60)
    print("  Decision Tree — Attack_type sınıflandırma (multi-class)")
    print("=" * 60)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode}")

    spark = get_spark("Model-2-DecisionTree-Multiclass")

    print("\n[1/7] Multi-class veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols, label_index_model = run_ml_pipeline_multiclass(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    label_names = list(label_index_model.labels)
    num_classes = len(label_names)
    print(f"   ⏱️ Pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature: {len(feature_cols)} | Sınıf: {num_classes}")

    print("\n[2/7] Decision Tree pipeline kuruluyor...")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxDepth=10,
        seed=42,
    )
    pipeline = Pipeline(stages=[dt])

    print("\n[3/7] Cross Validation...")
    if args.fast:
        max_depth_values = [5, 10]
        min_instances_values = [1, 5]
        impurity_values = ["gini"]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            max_depth_values = [5, 10, 15, 20]
            min_instances_values = [1, 5, 10]
            impurity_values = ["gini", "entropy"]
            num_folds = 3
        else:
            max_depth_values = [5, 10, 15]
            min_instances_values = [1, 5]
            impurity_values = ["gini", "entropy"]
            num_folds = 2

    param_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, max_depth_values)
        .addGrid(dt.minInstancesPerNode, min_instances_values)
        .addGrid(dt.impurity, impurity_values)
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

    total_cv_runs = len(param_grid) * num_folds
    print(f"   📌 Grid: {len(param_grid)} | Fold: {num_folds} | Toplam fit: {total_cv_runs}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    print(f"   ⏱️ CV süresi: {time.perf_counter() - cv_start:.2f}s")

    best_dt_model = cv_model.bestModel.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"maxDepth={best_dt_model.getOrDefault('maxDepth')}, "
        f"minInstancesPerNode={best_dt_model.getOrDefault('minInstancesPerNode')}, "
        f"impurity={best_dt_model.getOrDefault('impurity')}"
    )

    print("\n[4/7] Test seti değerlendirme...")
    eval_start = time.perf_counter()
    predictions = cv_model.bestModel.transform(test_df)
    metrics = evaluate_model_multiclass(predictions, num_classes=num_classes)
    confusion = compute_confusion_matrix_multiclass(predictions, label_names=label_names)
    print(f"   ⏱️ Test süresi: {time.perf_counter() - eval_start:.2f}s")

    print("\n[5/7] Ağaç yapısı analiz...")
    tree_info = analyze_tree_structure(cv_model)

    print("\n[6/7] Feature importance...")
    top_features = extract_feature_importance(cv_model, feature_cols)
    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")
    chart_path = "/opt/bitnami/spark/ml/dt_feature_importance.png"
    save_feature_importance_chart(top_features, chart_path)

    print("\n[7/7] MLflow'a loglanıyor...")
    best_params = {
        "maxDepth": int(best_dt_model.getOrDefault("maxDepth")),
        "minInstancesPerNode": int(best_dt_model.getOrDefault("minInstancesPerNode")),
        "impurity": str(best_dt_model.getOrDefault("impurity")),
        "numClasses": int(num_classes),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "tree_depth": tree_info["depth"],
        "tree_num_nodes": tree_info["num_nodes"],
        "tree_num_leaves": tree_info["num_leaves"],
        "label_column": "Attack_type",
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    metrics["tree_depth"] = float(tree_info["depth"])
    metrics["tree_num_nodes"] = float(tree_info["num_nodes"])

    confusion_metrics_for_mlflow = {}
    for i, name in enumerate(label_names):
        confusion_metrics_for_mlflow[f"row_total_class_{i}"] = confusion["row_totals"][i]
        confusion_metrics_for_mlflow[f"per_class_acc_{i}"] = confusion["per_class_acc"][i]

    run_name = "decision_tree_v1_fast" if args.fast else "decision_tree_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="DecisionTree",
        params=best_params,
        metrics={**metrics, **confusion_metrics_for_mlflow},
        model=cv_model.bestModel,
        confusion_matrix=None,
        feature_importance=top_features,
        tags={
            "task": "step_6_3",
            "model_index": "2",
            "classification_type": "multiclass",
            "label_column": "Attack_type",
            "num_classes": str(num_classes),
            "interpretable": "true",
        },
    )

    try:
        import mlflow
        import os
        with mlflow.start_run(run_id=run_id):
            mlflow.log_text(tree_info["debug_string_preview"], "tree_structure.txt")
            mlflow.log_text(confusion_matrix_to_text(confusion), "confusion_matrix.csv")
            mlflow.log_text(
                "\n".join(f"{i},{name}" for i, name in enumerate(label_names)),
                "label_index_mapping.csv",
            )
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, "charts")
    except Exception as e:
        print(f"   ⚠️ Artifact loglama hatası: {e}")

    print("\n" + "=" * 60)
    print("  ✅ Decision Tree (Multi-class) tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Sınıf sayısı: {num_classes}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  Ağaç derinliği: {tree_info['depth']} | Düğüm: {tree_info['num_nodes']}")
    print(f"  Toplam süre: {time.perf_counter() - total_start:.2f}s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
