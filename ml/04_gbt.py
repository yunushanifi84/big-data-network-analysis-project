"""
Adım 6.5 — Model 4: Gradient Boosted Trees (GBT)
===================================================
En güçlü model adayı olarak GBT eğitir, CrossValidator ile optimize eder
ve sonuçları MLflow'a loglar. Random Forest ile karşılaştırma yapar.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/04_gbt.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/04_gbt.py --fast --sample-size 5000
"""

import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
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
    parser = argparse.ArgumentParser(description="Adım 6.5 GBT eğitimi")
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
#  Feature Importance
# ─────────────────────────────────────────────

def extract_feature_importance(cv_model, feature_cols):
    """Best model'in featureImportances'ından top 10 feature çıkarır."""
    best_pipeline_model = cv_model.bestModel
    gbt_model = best_pipeline_model.stages[-1]
    importances = gbt_model.featureImportances.toArray().tolist()

    pairs = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    return pairs[:10]


def extract_all_feature_importance(cv_model, feature_cols):
    """Tüm feature importance skorlarını döndürür."""
    best_pipeline_model = cv_model.bestModel
    gbt_model = best_pipeline_model.stages[-1]
    importances = gbt_model.featureImportances.toArray().tolist()

    return sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )


# ─────────────────────────────────────────────
#  CV Sonuç Analizi
# ─────────────────────────────────────────────

def analyze_cv_results(cv_model, param_grid):
    """CrossValidator sonuçlarını analiz eder ve raporlar."""
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
        "all_scores": [float(s) for s in avg_metrics],
        "details": cv_details,
    }


# ─────────────────────────────────────────────
#  GBT Ensemble Analizi
# ─────────────────────────────────────────────

def analyze_gbt_structure(cv_model):
    """GBT ensemble yapısını analiz eder."""
    gbt_model = cv_model.bestModel.stages[-1]

    num_trees = gbt_model.getNumTrees
    total_nodes = gbt_model.totalNumNodes
    tree_weights = list(gbt_model.treeWeights)

    tree_depths = [t.depth for t in gbt_model.trees]
    tree_nodes = [t.numNodes for t in gbt_model.trees]
    avg_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0

    print("\n🌲 GBT Ensemble Yapı Analizi:")
    print(f"   İterasyon (ağaç) sayısı:      {num_trees}")
    print(f"   Toplam düğüm sayısı:          {total_nodes}")
    print(f"   Ortalama ağaç derinliği:      {avg_depth:.1f}")
    print(f"   Min / Max ağaç derinliği:     {min(tree_depths)} / {max(tree_depths)}")
    print(f"   Ağaç ağırlıkları (ilk 5):     {[round(w, 4) for w in tree_weights[:5]]}")

    return {
        "num_trees": num_trees,
        "total_nodes": total_nodes,
        "avg_depth": round(avg_depth, 2),
        "max_depth": max(tree_depths),
        "min_depth": min(tree_depths),
        "tree_weights_sample": [round(w, 4) for w in tree_weights[:10]],
    }


# ─────────────────────────────────────────────
#  RF ile Karşılaştırma
# ─────────────────────────────────────────────

def compare_with_rf(gbt_metrics):
    """
    MLflow'dan en son Random Forest run'ını bulup GBT ile karşılaştırır.
    Bulamazsa sadece uyarı verir.
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("iot_intrusion_detection")
        if experiment is None:
            print("   ⚠️ Experiment bulunamadı, karşılaştırma yapılamadı.")
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'RandomForest'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            print("   ⚠️ Random Forest run'ı bulunamadı, karşılaştırma yapılamadı.")
            return None

        rf_run = runs[0]
        rf_metrics = rf_run.data.metrics

        compare_keys = [
            ("auc_roc", "AUC-ROC"),
            ("accuracy", "Accuracy"),
            ("f1_score", "F1-Score"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]

        print("\n📊 GBT vs Random Forest Karşılaştırması:")
        print(f"   {'Metrik':<15} {'GBT':>10} {'RF':>10} {'Fark':>10} {'Kazanan':>10}")
        print(f"   {'─'*55}")

        comparison = {}
        for key, label in compare_keys:
            gbt_val = gbt_metrics.get(key, 0)
            rf_val = rf_metrics.get(key, 0)
            diff = gbt_val - rf_val
            winner = "GBT" if diff > 0 else ("RF" if diff < 0 else "Eşit")
            print(f"   {label:<15} {gbt_val:>10.4f} {rf_val:>10.4f} {diff:>+10.4f} {winner:>10}")
            comparison[key] = {"gbt": gbt_val, "rf": rf_val, "diff": diff}

        return comparison
    except Exception as e:
        print(f"   ⚠️ Karşılaştırma yapılamadı: {e}")
        return None


# ─────────────────────────────────────────────
#  Görselleştirme
# ─────────────────────────────────────────────

def save_feature_importance_chart(top_features, output_path):
    """En önemli feature'ları horizontal bar chart olarak kaydeder."""
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

        ax.set_xlabel("Feature Importance (GBT)", fontsize=11)
        ax.set_title("GBT — Top 10 Feature Importance", fontsize=13, fontweight="bold")
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
    """Tüm feature importance skorlarını CSV olarak kaydeder."""
    try:
        lines = ["rank,feature_name,importance"]
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
    print("  Adım 6.5 — Gradient Boosted Trees (GBT) Eğitimi")
    print("=" * 60)
    print(f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
          f"CV parallelism: {args.cv_parallelism}")

    # ── 1) Spark başlat ──
    spark = get_spark("Model-4-GBT")

    # ── 2) Ortak pipeline ──
    print("\n[1/9] Veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols = run_ml_pipeline(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    print(f"   ⏱️ Ortak pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature sayısı: {len(feature_cols)}")

    # ── 3) GBT + Pipeline ──
    print("\n[2/9] GBT pipeline kuruluyor...")
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42,
    )
    pipeline = Pipeline(stages=[gbt])

    # ── 4) Cross Validation ──
    print("\n[3/9] Cross Validation ile hiperparametre optimizasyonu...")
    if args.fast:
        max_iter_values = [30, 50]
        max_depth_values = [3, 5]
        step_size_values = [0.1]
        num_folds = 2
    else:
        if args.cv_mode == "full":
            max_iter_values = [30, 50, 100]
            max_depth_values = [3, 5, 7]
            step_size_values = [0.05, 0.1, 0.2]
            num_folds = 5
        else:
            max_iter_values = [30, 50]
            max_depth_values = [3, 5, 7]
            step_size_values = [0.1, 0.2]
            num_folds = 3

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, max_iter_values)
        .addGrid(gbt.maxDepth, max_depth_values)
        .addGrid(gbt.stepSize, step_size_values)
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
    print(f"   📌 maxIter: {max_iter_values}")
    print(f"   📌 maxDepth: {max_depth_values}")
    print(f"   📌 stepSize: {step_size_values}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    cv_duration = time.perf_counter() - cv_start
    print(f"   ⏱️ Cross Validation süresi: {cv_duration:.2f}s")

    best_pipeline_model = cv_model.bestModel
    best_gbt_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"maxIter={best_gbt_model.getOrDefault('maxIter')}, "
        f"maxDepth={best_gbt_model.getOrDefault('maxDepth')}, "
        f"stepSize={best_gbt_model.getOrDefault('stepSize')}"
    )

    # ── 5) CV sonuçları ──
    print("\n[4/9] Cross Validation sonuçları analiz ediliyor...")
    cv_results = analyze_cv_results(cv_model, param_grid)

    # ── 6) Test seti değerlendirme ──
    print("\n[5/9] Test seti üzerinde değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model(predictions)
    confusion = compute_confusion_matrix(predictions)
    print(f"   ⏱️ Test değerlendirme süresi: {time.perf_counter() - eval_start:.2f}s")

    # ── 7) GBT yapı analizi ──
    print("\n[6/9] GBT ensemble yapısı analiz ediliyor...")
    gbt_info = analyze_gbt_structure(cv_model)

    # ── 8) Feature importance ──
    print("\n[7/9] Feature importance çıkarılıyor...")
    fi_start = time.perf_counter()
    top_features = extract_feature_importance(cv_model, feature_cols)
    all_features = extract_all_feature_importance(cv_model, feature_cols)

    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")

    chart_path = "/opt/bitnami/spark/ml/gbt_feature_importance.png"
    save_feature_importance_chart(top_features, chart_path)

    csv_path = "/opt/bitnami/spark/ml/gbt_feature_importance.csv"
    save_feature_importance_csv(all_features, csv_path)
    print(f"   ⏱️ Feature importance süresi: {time.perf_counter() - fi_start:.2f}s")

    # ── 8b) RF ile karşılaştırma ──
    print("\n[8/9] Random Forest ile karşılaştırma yapılıyor...")
    rf_comparison = compare_with_rf(metrics)

    # ── 9) MLflow Logging ──
    print("\n[9/9] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()
    best_params = {
        "maxIter": int(best_gbt_model.getOrDefault("maxIter")),
        "maxDepth": int(best_gbt_model.getOrDefault("maxDepth")),
        "stepSize": float(best_gbt_model.getOrDefault("stepSize")),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "gbt_num_trees": gbt_info["num_trees"],
        "gbt_total_nodes": gbt_info["total_nodes"],
        "gbt_avg_depth": gbt_info["avg_depth"],
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    run_name = "gbt_v1_fast" if args.fast else "gbt_v1"

    # Ek metrikler
    metrics["cv_best_auc"] = cv_results["best_score"]
    metrics["gbt_num_trees"] = float(gbt_info["num_trees"])
    metrics["gbt_total_nodes"] = float(gbt_info["total_nodes"])

    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="GBTClassifier",
        params=best_params,
        metrics=metrics,
        model=best_pipeline_model,
        confusion_matrix=confusion,
        feature_importance=top_features,
        tags={
            "task": "step_6_5",
            "model_index": "4",
            "classification_type": "binary",
            "ensemble_method": "boosting",
            "production_candidate": "true",
        },
    )

    # Ek artifact'leri MLflow'a logla
    try:
        import mlflow
        import os

        with mlflow.start_run(run_id=run_id):
            # CV sonuçları CSV
            cv_csv = ["combination,avg_auc_roc,params"]
            for d in cv_results["details"]:
                cv_csv.append(f"{d['index']+1},{d['score']:.6f},{d['params']}")
            mlflow.log_text("\n".join(cv_csv) + "\n", "cv_results.csv")

            # Feature importance CSV
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, "feature_importance")

            # Feature importance chart
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, "charts")

            # GBT yapı özeti
            summary_lines = [
                "GBT Ensemble Yapı Özeti",
                "=" * 40,
                f"İterasyon sayısı:         {gbt_info['num_trees']}",
                f"Toplam düğüm:             {gbt_info['total_nodes']}",
                f"Ortalama derinlik:        {gbt_info['avg_depth']}",
                f"Min / Max derinlik:       {gbt_info['min_depth']} / {gbt_info['max_depth']}",
                f"Ağaç ağırlıkları (ilk 10): {gbt_info['tree_weights_sample']}",
            ]
            mlflow.log_text("\n".join(summary_lines) + "\n", "gbt_structure.txt")

            # RF karşılaştırma sonuçları
            if rf_comparison:
                comp_lines = ["metric,gbt,rf,diff"]
                for key, vals in rf_comparison.items():
                    comp_lines.append(
                        f"{key},{vals['gbt']:.6f},{vals['rf']:.6f},{vals['diff']:+.6f}"
                    )
                mlflow.log_text("\n".join(comp_lines) + "\n", "gbt_vs_rf_comparison.csv")

        print("   ✅ CV sonuçları, chart ve karşılaştırma MLflow'a loglandı.")
    except Exception as e:
        print(f"   ⚠️ Ek artifact'ler loglanamadı: {e}")

    print(f"   ⏱️ MLflow loglama süresi: {time.perf_counter() - mlflow_start:.2f}s")

    # ── Özet ──
    print("\n" + "=" * 60)
    print("  ✅ Adım 6.5 — GBT tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  MLflow UI: http://localhost:5000")
    print(f"  En iyi maxIter: {best_params['maxIter']}")
    print(f"  En iyi maxDepth: {best_params['maxDepth']}")
    print(f"  En iyi stepSize: {best_params['stepSize']}")
    print(f"  GBT — ağaç sayısı: {gbt_info['num_trees']}")
    print(f"  GBT — toplam düğüm: {gbt_info['total_nodes']}")
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
