"""
Adım 6.4 — Model 3: Random Forest (Multi-class)
==================================================
Saldırı tipi (Attack_type) için ensemble Random Forest.
CrossValidator ile hiperparametre seçer, MLflow'a loglar.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/03_random_forest.py
"""
import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
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
    parser = argparse.ArgumentParser(description="Random Forest multi-class eğitimi")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--cv-parallelism", type=int, default=2)
    return parser.parse_args()


def extract_feature_importance(cv_model, feature_cols, top_n=10):
    rf_model = cv_model.bestModel.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n], pairs


def analyze_forest_structure(cv_model):
    rf_model = cv_model.bestModel.stages[-1]
    num_trees = rf_model.getNumTrees
    total_nodes = rf_model.totalNumNodes
    depths = [t.depth for t in rf_model.trees]
    nodes = [t.numNodes for t in rf_model.trees]
    avg_depth = sum(depths) / len(depths) if depths else 0
    print("\n🌲 Random Forest Yapı:")
    print(f"   Ağaç: {num_trees} | Toplam düğüm: {total_nodes} | "
          f"Ort. derinlik: {avg_depth:.1f}")
    return {
        "num_trees": num_trees,
        "total_nodes": total_nodes,
        "avg_depth": round(avg_depth, 2),
        "max_depth": max(depths) if depths else 0,
        "min_depth": min(depths) if depths else 0,
        "tree_depths": depths,
        "tree_nodes": nodes,
    }


def analyze_cv_results(cv_model, param_grid):
    avg_metrics = cv_model.avgMetrics
    best_index = int(max(range(len(avg_metrics)), key=lambda i: avg_metrics[i]))
    best_score = float(avg_metrics[best_index])

    print("\n📊 CV Sonuçları (F1):")
    cv_details = []
    for i, (params, score) in enumerate(zip(param_grid, avg_metrics)):
        param_str = ", ".join(f"{p.name}={v}" for p, v in params.items())
        marker = " ← best" if i == best_index else ""
        print(f"   {i+1:<3} F1={score:.4f}  {param_str}{marker}")
        cv_details.append({"index": i, "score": float(score), "params": param_str})

    return {
        "best_score": best_score,
        "best_index": best_index,
        "details": cv_details,
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
        bars = ax.barh(names, values, color="#4CAF50", edgecolor="#2E7D32", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title("Random Forest — Top 10 Feature Importance (Multi-class)",
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

    print("=" * 60)
    print("  Random Forest — Attack_type sınıflandırma (multi-class)")
    print("=" * 60)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode}")

    spark = get_spark("Model-3-RandomForest-Multiclass")

    print("\n[1/8] Multi-class veri pipeline hazırlanıyor...")
    train_df, test_df, feature_cols, label_index_model = run_ml_pipeline_multiclass(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    label_names = list(label_index_model.labels)
    num_classes = len(label_names)
    print(f"   📌 Feature: {len(feature_cols)} | Sınıf: {num_classes}")

    print("\n[2/8] Random Forest pipeline kuruluyor...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=15,
        seed=42,
    )
    pipeline = Pipeline(stages=[rf])

    print("\n[3/8] Cross Validation...")
    if args.fast:
        num_trees_values = [50, 100]
        max_depth_values = [10, 15]
        min_instances_values = [1]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            num_trees_values = [50, 100, 200]
            max_depth_values = [10, 15, 20]
            min_instances_values = [1, 5]
            num_folds = 5
        else:
            num_trees_values = [50, 100]
            max_depth_values = [10, 15]
            min_instances_values = [1, 5]
            num_folds = 3

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, num_trees_values)
        .addGrid(rf.maxDepth, max_depth_values)
        .addGrid(rf.minInstancesPerNode, min_instances_values)
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

    best_rf_model = cv_model.bestModel.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"numTrees={best_rf_model.getNumTrees}, "
        f"maxDepth={best_rf_model.getOrDefault('maxDepth')}, "
        f"minInstancesPerNode={best_rf_model.getOrDefault('minInstancesPerNode')}"
    )

    print("\n[4/8] CV analiz...")
    cv_results = analyze_cv_results(cv_model, param_grid)

    print("\n[5/8] Test seti değerlendirme...")
    eval_start = time.perf_counter()
    predictions = cv_model.bestModel.transform(test_df)
    metrics = evaluate_model_multiclass(predictions, num_classes=num_classes)
    confusion = compute_confusion_matrix_multiclass(predictions, label_names=label_names)
    print(f"   ⏱️ Test süresi: {time.perf_counter() - eval_start:.2f}s")

    print("\n[6/8] Forest yapı analiz...")
    forest_info = analyze_forest_structure(cv_model)

    print("\n[7/8] Feature importance...")
    top_features, all_features = extract_feature_importance(cv_model, feature_cols)
    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")
    chart_path = "/opt/bitnami/spark/ml/rf_feature_importance.png"
    csv_path = "/opt/bitnami/spark/ml/rf_feature_importance.csv"
    save_feature_importance_chart(top_features, chart_path)
    save_feature_importance_csv(all_features, csv_path)

    print("\n[8/8] MLflow'a loglanıyor...")
    best_params = {
        "numTrees": int(best_rf_model.getNumTrees),
        "maxDepth": int(best_rf_model.getOrDefault("maxDepth")),
        "minInstancesPerNode": int(best_rf_model.getOrDefault("minInstancesPerNode")),
        "numClasses": int(num_classes),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "forest_total_nodes": forest_info["total_nodes"],
        "forest_avg_depth": forest_info["avg_depth"],
        "label_column": "Attack_type",
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    metrics["cv_best_f1"] = cv_results["best_score"]
    metrics["forest_num_trees"] = float(forest_info["num_trees"])
    metrics["forest_total_nodes"] = float(forest_info["total_nodes"])
    metrics["forest_avg_depth"] = float(forest_info["avg_depth"])

    confusion_metrics_for_mlflow = {}
    for i, name in enumerate(label_names):
        confusion_metrics_for_mlflow[f"row_total_class_{i}"] = confusion["row_totals"][i]
        confusion_metrics_for_mlflow[f"per_class_acc_{i}"] = confusion["per_class_acc"][i]

    run_name = "random_forest_v1_fast" if args.fast else "random_forest_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="RandomForest",
        params=best_params,
        metrics={**metrics, **confusion_metrics_for_mlflow},
        model=cv_model.bestModel,
        confusion_matrix=None,
        feature_importance=top_features,
        tags={
            "task": "step_6_4",
            "model_index": "3",
            "classification_type": "multiclass",
            "label_column": "Attack_type",
            "num_classes": str(num_classes),
            "ensemble_method": "bagging",
        },
    )

    try:
        import mlflow
        import os
        with mlflow.start_run(run_id=run_id):
            cv_csv = ["combination,avg_f1,params"]
            for d in cv_results["details"]:
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

    print("\n" + "=" * 60)
    print("  ✅ Random Forest (Multi-class) tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Sınıf sayısı: {num_classes}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  Toplam süre: {time.perf_counter() - total_start:.2f}s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
