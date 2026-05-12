"""
Adım 6.6 — Model 5: Naive Bayes
==================================
Hızlı ve basit bir baseline olarak Naive Bayes eğitir ve MLflow'a loglar.
PDF'de zorunlu 5. model.

Önemli:
  - Multinomial NB negatif feature kabul etmez → MinMaxScaler ile [0,1] normalize.
  - En hızlı eğitilen model; performansı diğerlerinden düşük olabilir.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/05_naive_bayes.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/05_naive_bayes.py --fast --sample-size 5000
"""

import argparse
import sys
import time
import math

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

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
    parser = argparse.ArgumentParser(description="Adım 6.6 Naive Bayes eğitimi")
    parser.add_argument("--fast", action="store_true",
                        help="Hızlı mod (örneklem ile sınırlar)")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Hızlı modda kullanılacak satır sayısı")
    parser.add_argument("--cv-mode", choices=["quick", "full"], default="quick",
                        help="CV kapsamı: quick veya full")
    parser.add_argument("--cv-parallelism", type=int, default=2,
                        help="CrossValidator parallelism değeri")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Negatif Değer Kontrolü & MinMaxScaler
# ─────────────────────────────────────────────

def check_and_fix_negatives(train_df, test_df):
    """
    Feature vektöründe negatif değer olup olmadığını kontrol eder.
    Varsa MinMaxScaler ile [0, 1] aralığına normalize eder.

    Returns:
        (train_df, test_df, scaler_model_or_None, used_scaler: bool)
    """
    from pyspark.ml.functions import vector_to_array

    print("   🔍 Negatif feature değerleri kontrol ediliyor...")

    # features vektöründen min değeri bul
    min_val_row = train_df.select(
        F.array_min(vector_to_array(F.col("features"))).alias("min_val")
    ).agg(F.min("min_val").alias("global_min")).collect()[0]

    global_min = float(min_val_row["global_min"])
    print(f"   📊 Feature global min değeri: {global_min:.4f}")

    if global_min < 0:
        print("   ⚠️ Negatif değerler tespit edildi → MinMaxScaler uygulanıyor...")

        scaler = MinMaxScaler(
            inputCol="features",
            outputCol="scaled_features",
            min=0.0,
            max=1.0,
        )
        scaler_model = scaler.fit(train_df)

        train_df = (
            scaler_model.transform(train_df)
            .drop("features")
            .withColumnRenamed("scaled_features", "features")
        )
        test_df = (
            scaler_model.transform(test_df)
            .drop("features")
            .withColumnRenamed("scaled_features", "features")
        )

        print("   ✅ MinMaxScaler uygulandı — tüm değerler [0, 1] aralığında.")
        return train_df, test_df, scaler_model, True
    else:
        print("   ✅ Negatif değer yok — normalizasyon gerekmedi.")
        return train_df, test_df, None, False


# ─────────────────────────────────────────────
#  Feature Discriminative Power (NB'ye özgü)
# ─────────────────────────────────────────────

def extract_feature_importance_nb(nb_model, feature_cols):
    """
    Naive Bayes'te doğrudan featureImportances yoktur.
    Class log-probability farkını (|theta_class0 - theta_class1|) kullanarak
    her feature'ın sınıf ayrım gücünü hesaplar.

    Returns:
        list: [(feature_name, discriminative_power)] — top 10
    """
    try:
        # theta: class log-probabilities, shape (numClasses, numFeatures)
        theta = nb_model.theta.toArray()  # numpy array

        if theta.shape[0] < 2:
            print("   ⚠️ Tek sınıf — feature importance hesaplanamadı.")
            return [(f, 0.0) for f in feature_cols[:10]]

        # İki sınıf arasındaki log-probability farkının mutlak değeri
        diff = [abs(float(theta[0][j] - theta[1][j])) for j in range(theta.shape[1])]

        pairs = sorted(
            zip(feature_cols, diff), key=lambda x: x[1], reverse=True
        )
        return pairs[:10]
    except Exception as e:
        print(f"   ⚠️ Feature importance hesaplanamadı: {e}")
        # Fallback: pi (class prior) bazlı basit analiz
        return [(f, 0.0) for f in feature_cols[:10]]


def extract_all_feature_importance_nb(nb_model, feature_cols):
    """Tüm feature'ların discriminative power skorlarını döndürür."""
    try:
        theta = nb_model.theta.toArray()
        if theta.shape[0] < 2:
            return [(f, 0.0) for f in feature_cols]

        diff = [abs(float(theta[0][j] - theta[1][j])) for j in range(theta.shape[1])]
        return sorted(zip(feature_cols, diff), key=lambda x: x[1], reverse=True)
    except Exception:
        return [(f, 0.0) for f in feature_cols]


# ─────────────────────────────────────────────
#  NB Model Analizi
# ─────────────────────────────────────────────

def analyze_nb_model(nb_model):
    """Naive Bayes model iç yapısını analiz eder."""
    try:
        pi = nb_model.pi.toArray().tolist()   # class log-priors
        theta = nb_model.theta.toArray()       # class log-likelihoods

        print("\n📊 Naive Bayes Model Analizi:")
        print(f"   Model tipi:          {nb_model.getModelType()}")
        print(f"   Smoothing:           {nb_model.getSmoothing()}")
        print(f"   Sınıf sayısı:        {len(pi)}")
        print(f"   Feature sayısı:      {theta.shape[1]}")
        print(f"   Class log-priors (pi):")
        for i, p in enumerate(pi):
            label_name = "Normal" if i == 0 else "Attack"
            print(f"      {label_name} (class={i}): log(π)={p:.4f}, π={math.exp(p):.4f}")

        return {
            "model_type": nb_model.getModelType(),
            "smoothing": nb_model.getSmoothing(),
            "num_classes": len(pi),
            "num_features": int(theta.shape[1]),
            "class_priors": [round(math.exp(p), 4) for p in pi],
            "class_log_priors": [round(p, 4) for p in pi],
        }
    except Exception as e:
        print(f"   ⚠️ Model analizi yapılamadı: {e}")
        return {"model_type": "unknown", "smoothing": 1.0}


# ─────────────────────────────────────────────
#  CV Sonuç Analizi
# ─────────────────────────────────────────────

def analyze_cv_results(cv_model, param_grid):
    """CrossValidator sonuçlarını analiz eder."""
    avg_metrics = cv_model.avgMetrics
    best_index = int(max(range(len(avg_metrics)), key=lambda i: avg_metrics[i]))
    best_score = float(avg_metrics[best_index])

    print("\n📊 Cross Validation Sonuçları (AUC-ROC):")
    print(f"   {'#':<4} {'Ortalama AUC':<15} {'Parametreler'}")
    print(f"   {'─'*60}")

    cv_details = []
    for i, (params, score) in enumerate(zip(param_grid, avg_metrics)):
        param_str = ", ".join(f"{p.name}={v}" for p, v in params.items())
        marker = " ← best" if i == best_index else ""
        print(f"   {i+1:<4} {score:<15.6f} {param_str}{marker}")
        cv_details.append({"index": i, "score": float(score), "params": param_str})

    print(f"\n   ✅ En iyi CV skoru: {best_score:.6f} (kombinasyon #{best_index + 1})")
    return {
        "best_score": best_score,
        "best_index": best_index,
        "details": cv_details,
    }


# ─────────────────────────────────────────────
#  Görselleştirme
# ─────────────────────────────────────────────

def save_feature_importance_chart(top_features, output_path):
    """En etkili feature'ları horizontal bar chart olarak kaydeder."""
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

        ax.set_xlabel("Discriminative Power (|Δ log-probability|)", fontsize=11)
        ax.set_title("Naive Bayes — Top 10 Feature Discriminative Power",
                      fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Feature importance chart kaydedildi: {output_path}")
    except ImportError:
        print("   ⚠️ matplotlib yüklü değil, chart kaydedilemedi.")
    except Exception as e:
        print(f"   ⚠️ Chart kaydedilemedi: {e}")


def save_feature_importance_csv(all_features, output_path):
    """Tüm feature discriminative power skorlarını CSV olarak kaydeder."""
    try:
        lines = ["rank,feature_name,discriminative_power"]
        for rank, (fname, imp) in enumerate(all_features, start=1):
            lines.append(f"{rank},{fname},{imp:.8f}")
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"   ✅ Feature importance CSV kaydedildi: {output_path}")
    except Exception as e:
        print(f"   ⚠️ CSV kaydedilemedi: {e}")


# ─────────────────────────────────────────────
#  Ana Akış
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 60)
    print("  Adım 6.6 — Naive Bayes Eğitimi")
    print("=" * 60)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
          f"CV parallelism: {args.cv_parallelism}")

    # ── 1) Spark başlat ──
    spark = get_spark("Model-5-NaiveBayes")

    # ── 2) Ortak pipeline ──
    print("\n[1/7] Veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols = run_ml_pipeline(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    print(f"   ⏱️ Ortak pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature sayısı: {len(feature_cols)}")

    # ── 3) Naive Bayes + MinMaxScaler Pipeline ──
    print("\n[2/7] Naive Bayes + MinMaxScaler pipeline kuruluyor...")
    scaler = MinMaxScaler(
        inputCol="features",
        outputCol="scaled_features",
        min=0.0,
        max=1.0,
    )
    nb = NaiveBayes(
        featuresCol="scaled_features",
        labelCol="label",
        modelType="multinomial",
        smoothing=1.0,
    )
    # NaiveBayes weightCol desteklemez — classWeight kullanılmaz
    # MinMaxScaler her CV fold'unda yeniden fit edilir (veri sızıntısı olmaz)
    pipeline = Pipeline(stages=[scaler, nb])

    # ── 4) Cross Validation ──
    print("\n[3/7] Cross Validation ile hiperparametre optimizasyonu...")

    # Spark 3.x'te sadece "multinomial" ve "complement" desteklenir;
    # "gaussian" ≥ 3.0'da yok. Güvenli liste:
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

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC",
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=42,
        parallelism=args.cv_parallelism,
    )

    total_cv_runs = len(param_grid) * num_folds
    print(f"   📌 Grid boyutu: {len(param_grid)} kombinasyon | Fold: {num_folds} | "
          f"Toplam fit: {total_cv_runs}")
    print(f"   📌 smoothing: {smoothing_values}")
    print(f"   📌 modelType: {model_type_values}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    cv_duration = time.perf_counter() - cv_start
    print(f"   ⏱️ Cross Validation süresi: {cv_duration:.2f}s")

    best_pipeline_model = cv_model.bestModel
    best_nb_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"modelType={best_nb_model.getModelType()}, "
        f"smoothing={best_nb_model.getSmoothing()}"
    )

    # ── 5) CV sonuçları ──
    print("\n[4/7] Cross Validation sonuçları analiz ediliyor...")
    cv_results = analyze_cv_results(cv_model, param_grid)

    # ── 6) Test seti değerlendirme ──
    print("\n[5/7] Test seti üzerinde değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model(predictions)
    confusion = compute_confusion_matrix(predictions)
    print(f"   ⏱️ Test değerlendirme süresi: {time.perf_counter() - eval_start:.2f}s")

    # ── 7) NB model analizi ──
    print("\n[6/7] Naive Bayes model analizi ve feature importance...")
    fi_start = time.perf_counter()

    nb_info = analyze_nb_model(best_nb_model)
    top_features = extract_feature_importance_nb(best_nb_model, feature_cols)
    all_features = extract_all_feature_importance_nb(best_nb_model, feature_cols)

    print("\n   Top 10 discriminative feature:")
    for idx, (fname, power) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {power:.6f}")

    chart_path = "/opt/bitnami/spark/ml/nb_feature_importance.png"
    save_feature_importance_chart(top_features, chart_path)

    csv_path = "/opt/bitnami/spark/ml/nb_feature_importance.csv"
    save_feature_importance_csv(all_features, csv_path)
    print(f"   ⏱️ Feature importance süresi: {time.perf_counter() - fi_start:.2f}s")

    # ── 8) MLflow Logging ──
    print("\n[7/7] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()
    best_params = {
        "modelType": str(best_nb_model.getModelType()),
        "smoothing": float(best_nb_model.getSmoothing()),
        "used_minmax_scaler": True,
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    run_name = "naive_bayes_v1_fast" if args.fast else "naive_bayes_v1"

    metrics["cv_best_auc"] = cv_results["best_score"]

    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="NaiveBayes",
        params=best_params,
        metrics=metrics,
        model=best_pipeline_model,
        confusion_matrix=confusion,
        feature_importance=top_features,
        tags={
            "task": "step_6_6",
            "model_index": "5",
            "classification_type": "binary",
            "baseline_model": "true",
        },
    )

    # Ek artifact'ler
    try:
        import mlflow
        import os

        with mlflow.start_run(run_id=run_id):
            # CV sonuçları
            cv_csv = ["combination,avg_auc_roc,params"]
            for d in cv_results["details"]:
                cv_csv.append(f"{d['index']+1},{d['score']:.6f},{d['params']}")
            mlflow.log_text("\n".join(cv_csv) + "\n", "cv_results.csv")

            # Feature importance dosyaları
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, "feature_importance")
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, "charts")

            # NB model özeti
            summary = [
                "Naive Bayes Model Özeti",
                "=" * 40,
                f"Model tipi:       {nb_info.get('model_type', 'N/A')}",
                f"Smoothing:        {nb_info.get('smoothing', 'N/A')}",
                f"Sınıf sayısı:     {nb_info.get('num_classes', 'N/A')}",
                f"Feature sayısı:   {nb_info.get('num_features', 'N/A')}",
                f"MinMaxScaler:     Evet (Pipeline içinde, fold başına fit)",
                f"Class priors:     {nb_info.get('class_priors', 'N/A')}",
                f"Class log-priors: {nb_info.get('class_log_priors', 'N/A')}",
            ]
            mlflow.log_text("\n".join(summary) + "\n", "nb_model_summary.txt")

        print("   ✅ CV sonuçları, chart ve model özeti MLflow'a loglandı.")
    except Exception as e:
        print(f"   ⚠️ Ek artifact'ler loglanamadı: {e}")

    print(f"   ⏱️ MLflow loglama süresi: {time.perf_counter() - mlflow_start:.2f}s")

    # ── Özet ──
    print("\n" + "=" * 60)
    print("  ✅ Adım 6.6 — Naive Bayes tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  MLflow UI: http://localhost:5000")
    print(f"  Model tipi: {nb_info.get('model_type', 'N/A')}")
    print(f"  Smoothing: {nb_info.get('smoothing', 'N/A')}")
    print(f"  MinMaxScaler kullanıldı: Evet (Pipeline içinde)")
    print(f"  CV en iyi AUC: {cv_results['best_score']:.4f}")
    print(f"  AUC-ROC (test): {metrics.get('auc_roc', 0):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  Toplam eğitim süresi: {time.perf_counter() - total_start:.2f}s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
