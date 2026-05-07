"""
Adım 6.2 — Model 1: Logistic Regression
========================================
Binary sınıflandırma (Normal=0, Attack=1) için baseline model eğitir,
cross validation ile hiperparametre seçer ve sonuçları MLflow'a loglar.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/01_logistic_regression.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/01_logistic_regression.py --fast --sample-size 5000
"""

import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Proje kökünü Python path'e ekle
sys.path.insert(0, "/opt/bitnami/spark")

from spark.spark_session import get_spark
from ml.utils import (
    run_ml_pipeline,
    evaluate_model,
    compute_confusion_matrix,
    log_to_mlflow,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Adım 6.2 Logistic Regression eğitimi")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Hızlı mod (eğitim verisini örneklem ile sınırlar)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Hızlı modda train setinden kullanılacak satır sayısı",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["quick", "full"],
        default="quick",
        help="CV kapsamı: quick (daha hızlı) veya full (daha kapsamlı)",
    )
    parser.add_argument(
        "--cv-parallelism",
        type=int,
        default=2,
        help="CrossValidator parallelism değeri",
    )
    return parser.parse_args()


def extract_feature_importance(cv_model, feature_cols):
    """
    Best model katsayılarından en önemli 10 feature'ı çıkarır.
    """
    best_pipeline_model = cv_model.bestModel
    lr_model = best_pipeline_model.stages[-1]
    coefficients = lr_model.coefficients.toArray().tolist()

    importance_pairs = [
        (feature_name, abs(coef_value))
        for feature_name, coef_value in zip(feature_cols, coefficients)
    ]
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    return importance_pairs[:10]


def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 60)
    print("  Adım 6.2 — Logistic Regression Eğitimi")
    print("=" * 60)
    print(
        f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
        f"CV parallelism: {args.cv_parallelism}"
    )

    # 1) Spark başlat
    spark = get_spark("Model-1-LogisticRegression")

    # 2) Ortak pipeline (6.1 çıktıları)
    print("\n[1/6] Veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols = run_ml_pipeline(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    print(f"   ⏱️ Ortak pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature sayısı: {len(feature_cols)}")

    # 3) Logistic Regression + Pipeline
    print("\n[2/6] Logistic Regression pipeline kuruluyor...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=100,
        regParam=0.01,
    )
    pipeline = Pipeline(stages=[lr])

    # 4) Cross Validation
    print("\n[3/6] Cross Validation çalıştırılıyor...")
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

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        ),
        numFolds=num_folds,
        seed=42,
        parallelism=args.cv_parallelism,
    )

    total_cv_runs = len(param_grid) * num_folds
    print(
        f"   📌 Grid boyutu: {len(param_grid)} kombinasyon | Fold: {num_folds} | "
        f"Toplam fit: {total_cv_runs}"
    )
    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    print(f"   ⏱️ Cross Validation süresi: {time.perf_counter() - cv_start:.2f}s")
    best_pipeline_model = cv_model.bestModel
    best_lr_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"regParam={best_lr_model.getRegParam()}, "
        f"elasticNetParam={best_lr_model.getElasticNetParam()}, "
        f"maxIter={best_lr_model.getMaxIter()}"
    )

    # 5) Test seti değerlendirme
    print("\n[4/6] Test seti üzerinde değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model(predictions)
    confusion = compute_confusion_matrix(predictions)
    print(f"   ⏱️ Test değerlendirme süresi: {time.perf_counter() - eval_start:.2f}s")

    # 6) Feature importance + MLflow
    print("\n[5/6] Feature importance çıkarılıyor...")
    fi_start = time.perf_counter()
    top_features = extract_feature_importance(cv_model, feature_cols)
    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")
    print(f"   ⏱️ Feature importance süresi: {time.perf_counter() - fi_start:.2f}s")

    print("\n[6/6] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()
    best_params = {
        "maxIter": int(best_lr_model.getMaxIter()),
        "regParam": float(best_lr_model.getRegParam()),
        "elasticNetParam": float(best_lr_model.getElasticNetParam()),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    run_name = "logistic_regression_v1_fast" if args.fast else "logistic_regression_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="LogisticRegression",
        params=best_params,
        metrics=metrics,
        model=best_pipeline_model,
        confusion_matrix=confusion,
        feature_importance=top_features,
        tags={
            "task": "step_6_2",
            "model_index": "1",
            "classification_type": "binary",
        },
    )
    print(f"   ⏱️ MLflow loglama süresi: {time.perf_counter() - mlflow_start:.2f}s")

    print("\n" + "=" * 60)
    print("  ✅ Adım 6.2 tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print("  MLflow UI: http://localhost:5000")
    print(f"  Toplam eğitim süresi: {time.perf_counter() - total_start:.2f}s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
