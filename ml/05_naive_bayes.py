"""
Adım 6.6 — Model 5: Naive Bayes (Multi-class)
==================================================
Saldırı tipi (Attack_type) için multinomial Naive Bayes.
MinMaxScaler ile feature'ları [0,1] aralığına alır (NB negatif değer kabul etmez).

NOT: NaiveBayes Spark MLlib'de weightCol parametresi DESTEKLEMİYOR,
bu yüzden classWeight sütununu kullanamıyoruz. Sınıf dengesizliği
modelin doğal smoothing'i ile telafi edilir.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/05_naive_bayes.py
"""
import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import MinMaxScaler
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
    parser = argparse.ArgumentParser(description="Naive Bayes multi-class eğitimi")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--cv-parallelism", type=int, default=2)
    return parser.parse_args()


def extract_feature_importance_nb(nb_model, feature_cols, num_classes):
    """
    NB'de doğrudan featureImportances yoktur. theta matrix'inden
    (numClasses × numFeatures) her feature için sınıflar arası
    log-likelihood varyansını "discriminative power" olarak kullanırız.
    Yüksek varyans → feature sınıfları daha güçlü ayırıyor demektir.
    """
    try:
        theta = nb_model.theta.toArray()  # shape: (numClasses, numFeatures)
        n_classes_actual, n_features = theta.shape
        if n_classes_actual < 2:
            return [(f, 0.0) for f in feature_cols[:10]], []

        # Her feature için sınıflar arası varyans (discriminative power)
        scores = []
        for j in range(n_features):
            col_vals = theta[:, j]
            mean = sum(col_vals) / len(col_vals)
            var = sum((v - mean) ** 2 for v in col_vals) / len(col_vals)
            scores.append((feature_cols[j], float(var)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:10], scores
    except Exception as e:
        print(f"   ⚠️ Feature importance hesaplanamadı: {e}")
        return [(f, 0.0) for f in feature_cols[:10]], []


def analyze_nb_model(nb_model):
    import math
    try:
        pi = nb_model.pi.toArray().tolist()
        theta = nb_model.theta.toArray()
        print("\n📊 Naive Bayes Model Analizi:")
        print(f"   Model tipi:     {nb_model.getModelType()}")
        print(f"   Smoothing:      {nb_model.getSmoothing()}")
        print(f"   Sınıf sayısı:   {len(pi)}")
        print(f"   Feature sayısı: {theta.shape[1]}")
        return {
            "model_type": nb_model.getModelType(),
            "smoothing": nb_model.getSmoothing(),
            "num_classes": len(pi),
            "num_features": int(theta.shape[1]),
            "class_priors": [round(math.exp(p), 6) for p in pi],
        }
    except Exception as e:
        print(f"   ⚠️ Analiz başarısız: {e}")
        return {}


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
        bars = ax.barh(names, values, color="#9C27B0", edgecolor="#6A1B9A", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Discriminative Power (var of log-likelihoods)", fontsize=11)
        ax.set_title("Naive Bayes — Top 10 Feature (Multi-class)",
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
        lines = ["rank,feature_name,discriminative_power"]
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
    print("  Naive Bayes — Attack_type sınıflandırma (multi-class)")
    print("=" * 60)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode}")

    spark = get_spark("Model-5-NaiveBayes-Multiclass")

    print("\n[1/7] Multi-class veri pipeline hazırlanıyor...")
    train_df, test_df, feature_cols, label_index_model = run_ml_pipeline_multiclass(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    label_names = list(label_index_model.labels)
    num_classes = len(label_names)
    print(f"   📌 Feature: {len(feature_cols)} | Sınıf: {num_classes}")

    print("\n[2/7] Naive Bayes + MinMaxScaler pipeline kuruluyor...")
    scaler = MinMaxScaler(
        inputCol="features", outputCol="scaled_features",
        min=0.0, max=1.0,
    )
    nb = NaiveBayes(
        featuresCol="scaled_features",
        labelCol="label",
        modelType="multinomial",
        smoothing=1.0,
    )
    # NaiveBayes weightCol DESTEKLEMİYOR — classWeight kullanılmaz
    pipeline = Pipeline(stages=[scaler, nb])

    print("\n[3/7] Cross Validation...")
    if args.fast:
        smoothing_values = [0.5, 1.0]
        model_type_values = ["multinomial"]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            smoothing_values = [0.5, 1.0, 2.0]
            model_type_values = ["multinomial", "complement"]
            num_folds = 5
        else:
            smoothing_values = [0.5, 1.0, 2.0]
            model_type_values = ["multinomial"]
            num_folds = 3

    param_grid = (
        ParamGridBuilder()
        .addGrid(nb.smoothing, smoothing_values)
        .addGrid(nb.modelType, model_type_values)
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

    best_nb_model = cv_model.bestModel.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"modelType={best_nb_model.getModelType()}, smoothing={best_nb_model.getSmoothing()}"
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
    best_idx = int(max(range(len(avg_metrics)), key=lambda i: avg_metrics[i]))
    for i, (params, score) in enumerate(zip(param_grid, avg_metrics)):
        ps = ", ".join(f"{p.name}={v}" for p, v in params.items())
        marker = " ← best" if i == best_idx else ""
        print(f"   {i+1:<3} F1={score:.4f}  {ps}{marker}")
        cv_details.append({"index": i, "score": float(score), "params": ps})
    cv_best_f1 = float(max(avg_metrics)) if avg_metrics else 0.0

    print("\n[6/7] Model analiz + feature importance...")
    nb_info = analyze_nb_model(best_nb_model)
    top_features, all_features = extract_feature_importance_nb(
        best_nb_model, feature_cols, num_classes
    )
    print("   Top 10 discriminative feature:")
    for idx, (fname, power) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {power:.6f}")
    chart_path = "/opt/bitnami/spark/ml/nb_feature_importance.png"
    csv_path = "/opt/bitnami/spark/ml/nb_feature_importance.csv"
    save_feature_importance_chart(top_features, chart_path)
    save_feature_importance_csv(all_features, csv_path)

    print("\n[7/7] MLflow'a loglanıyor...")
    best_params = {
        "modelType": str(best_nb_model.getModelType()),
        "smoothing": float(best_nb_model.getSmoothing()),
        "used_minmax_scaler": True,
        "numClasses": int(num_classes),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
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

    run_name = "naive_bayes_v1_fast" if args.fast else "naive_bayes_v1"
    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="NaiveBayes",
        params=best_params,
        metrics={**metrics, **confusion_metrics_for_mlflow},
        model=cv_model.bestModel,
        confusion_matrix=None,
        feature_importance=top_features,
        tags={
            "task": "step_6_6",
            "model_index": "5",
            "classification_type": "multiclass",
            "label_column": "Attack_type",
            "num_classes": str(num_classes),
            "baseline_model": "true",
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

    print("\n" + "=" * 60)
    print("  ✅ Naive Bayes (Multi-class) tamamlandı!")
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
