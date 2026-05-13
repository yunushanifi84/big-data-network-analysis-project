"""
Adım 6.2 — Model 1: Multinomial Logistic Regression (Multi-class)
==================================================================
Saldırı **tipi** sınıflandırması (Attack_type kolonu) için multinomial
Logistic Regression eğitir, cross validation ile hiperparametre seçer ve
sonuçları MLflow'a loglar.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/01_logistic_regression.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/01_logistic_regression.py --fast --sample-size 30000
"""

import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
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
    parser = argparse.ArgumentParser(
        description="Multinomial Logistic Regression eğitimi (Attack_type)"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Hızlı mod (örneklem ile sınırlar)")
    parser.add_argument("--sample-size", type=int, default=30000,
                        help="Hızlı modda kullanılacak satır sayısı")
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick",
                        help="CV kapsamı")
    parser.add_argument("--cv-parallelism", type=int, default=2,
                        help="CrossValidator parallelism değeri")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="LR maxIter değeri")
    return parser.parse_args()


def extract_feature_importance(cv_model, feature_cols, label_names):
    """
    Multinomial LR için feature importance:
    coefficientMatrix shape = (numClasses, numFeatures).
    Her feature için sınıflar arası |katsayı| ortalaması = global önem.
    """
    best_pipeline_model = cv_model.bestModel
    lr_model = best_pipeline_model.stages[-1]

    coef_matrix = lr_model.coefficientMatrix.toArray()
    abs_matrix = [[abs(v) for v in row] for row in coef_matrix]

    n_classes = len(abs_matrix)
    n_features = len(feature_cols)

    global_importance = []
    for j in range(n_features):
        col_vals = [abs_matrix[i][j] for i in range(n_classes)]
        global_importance.append((feature_cols[j], sum(col_vals) / max(n_classes, 1)))
    global_importance.sort(key=lambda x: x[1], reverse=True)
    top_global = global_importance[:10]

    per_class_top = {}
    for i in range(n_classes):
        pairs = [(feature_cols[j], abs_matrix[i][j]) for j in range(n_features)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        cls_name = label_names[i] if i < len(label_names) else f"class_{i}"
        per_class_top[cls_name] = pairs[:5]

    return top_global, per_class_top


def confusion_matrix_to_text(cm: dict) -> str:
    labels = cm["labels"]
    matrix = cm["matrix"]
    lines = ["true_label," + ",".join(labels)]
    for i, name in enumerate(labels):
        row = ",".join(str(v) for v in matrix[i])
        lines.append(f"{name},{row}")
    return "\n".join(lines) + "\n"


def log_iteration_history_to_mlflow(run_id, lr_model):
    """LR eğitim sürecini MLflow'a loglar (objective, iteration metrikleri)."""
    try:
        import mlflow

        summary = lr_model.summary
        history = list(summary.objectiveHistory)
        if not history:
            return

        with mlflow.start_run(run_id=run_id):
            csv_lines = ["iteration,objective,objective_delta"]
            prev_obj = None
            for i, obj in enumerate(history):
                obj = float(obj)
                delta = 0.0 if prev_obj is None else obj - prev_obj
                mlflow.log_metric("objective", obj, step=i)
                mlflow.log_metric("objective_delta", float(delta), step=i)
                csv_lines.append(f"{i},{obj},{delta}")
                prev_obj = obj

            mlflow.log_metric("objective_final", float(history[-1]))
            mlflow.log_metric("objective_iterations", float(len(history)))
            mlflow.log_text("\n".join(csv_lines) + "\n", "objective_history.csv")

            for metric_name in ("accuracy", "weightedPrecision", "weightedRecall", "f1"):
                try:
                    v = getattr(summary, metric_name)
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"train_{metric_name}", float(v))
                except Exception:
                    continue
    except Exception as e:
        print(f"   ⚠️ Iterasyon logu yazılamadı: {e}")


def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 64)
    print("  Logistic Regression (Multinomial) — Attack_type sınıflandırma")
    print("=" * 64)
    print(
        f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
        f"parallelism: {args.cv_parallelism} | maxIter: {args.max_iter}"
    )

    spark = get_spark("Model-1-LogisticRegression-Multiclass")

    print("\n[1/6] Multi-class veri pipeline hazırlanıyor...")
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

    print("\n[2/6] Multinomial LR pipeline kuruluyor...")
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withMean=False, withStd=True,
    )
    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label",
        weightCol="classWeight",
        family="multinomial",
        maxIter=args.max_iter,
        regParam=0.01,
    )
    pipeline = Pipeline(stages=[scaler, lr])

    print("\n[3/6] Cross Validation...")
    if args.fast:
        reg_params = [0.01]
        elastic_net_params = [0.0]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            reg_params = [0.001, 0.01, 0.1]
            elastic_net_params = [0.0, 0.5, 1.0]
            num_folds = 3
        else:
            reg_params = [0.01, 0.1]
            elastic_net_params = [0.0, 0.5]
            num_folds = 2

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, reg_params)
        .addGrid(lr.elasticNetParam, elastic_net_params)
        .build()
    )

    cv_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1",
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=cv_evaluator,
        numFolds=num_folds,
        seed=42,
        parallelism=args.cv_parallelism,
    )

    total_cv_runs = len(param_grid) * num_folds
    print(f"   📌 Grid: {len(param_grid)} | Fold: {num_folds} | Toplam fit: {total_cv_runs}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    print(f"   ⏱️ CV süresi: {time.perf_counter() - cv_start:.2f}s")

    best_pipeline_model = cv_model.bestModel
    best_lr_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"regParam={best_lr_model.getRegParam()}, "
        f"elasticNetParam={best_lr_model.getElasticNetParam()}, "
        f"maxIter={best_lr_model.getMaxIter()}"
    )

    print("\n[4/6] Test seti değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model_multiclass(predictions, num_classes=num_classes)
    confusion = compute_confusion_matrix_multiclass(predictions, label_names=label_names)
    print(f"   ⏱️ Test süresi: {time.perf_counter() - eval_start:.2f}s")

    print("\n[5/6] Feature importance...")
    top_global, per_class_top = extract_feature_importance(
        cv_model, feature_cols, label_names
    )
    print("   Top 10 feature (sınıflar arası ortalama |katsayı|):")
    for idx, (fname, importance) in enumerate(top_global, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")

    print("\n[6/6] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()

    best_params = {
        "maxIter": int(best_lr_model.getMaxIter()),
        "regParam": float(best_lr_model.getRegParam()),
        "elasticNetParam": float(best_lr_model.getElasticNetParam()),
        "family": "multinomial",
        "numClasses": int(num_classes),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "label_column": "Attack_type",
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    cm_csv = confusion_matrix_to_text(confusion)

    confusion_metrics_for_mlflow = {}
    for i, name in enumerate(label_names):
        confusion_metrics_for_mlflow[f"row_total_class_{i}"] = confusion["row_totals"][i]
        confusion_metrics_for_mlflow[f"per_class_acc_{i}"] = confusion["per_class_acc"][i]

    run_name = "logistic_regression_v1_fast" if args.fast else "logistic_regression_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="LogisticRegression",
        params=best_params,
        metrics={**metrics, **confusion_metrics_for_mlflow},
        model=best_pipeline_model,
        confusion_matrix=None,
        feature_importance=top_global,
        tags={
            "task": "step_6_2",
            "model_index": "1",
            "classification_type": "multiclass",
            "label_column": "Attack_type",
            "num_classes": str(num_classes),
        },
    )

    try:
        import mlflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_text(cm_csv, "confusion_matrix.csv")
            mlflow.log_text(
                "\n".join(f"{i},{name}" for i, name in enumerate(label_names)),
                "label_index_mapping.csv",
            )
    except Exception as e:
        print(f"   ⚠️ Confusion matrix artifact yazılamadı: {e}")

    log_iteration_history_to_mlflow(run_id, best_lr_model)

    print(f"   ⏱️ MLflow süresi: {time.perf_counter() - mlflow_start:.2f}s")

    print("\n" + "=" * 64)
    print("  ✅ Logistic Regression (Multi-class) tamamlandı!")
    print("=" * 64)
    print(f"  Run ID: {run_id}")
    print(f"  Sınıf sayısı: {num_classes}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  Toplam süre: {time.perf_counter() - total_start:.2f}s")
    print("=" * 64)

    spark.stop()


if __name__ == "__main__":
    main()
