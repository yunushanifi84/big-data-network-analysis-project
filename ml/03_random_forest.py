"""
Adım 6.4 — Model 3: Random Forest
====================================
Ensemble yöntem olarak Random Forest eğitir, CrossValidator ile optimize eder
ve sonuçları MLflow'a loglar.

Avantajlar:
  - Ensemble: Birden fazla ağacın birleşimi ile daha kararlı tahminler.
  - Feature importance: Ağaçlar arası ortalama importance sağlar.
  - Decision Tree'ye göre overfitting riski daha düşük.

Not:
  - GBT ile karşılaştırılacak en güçlü model adaylarından biri.
  - Cross Validation zorunlu (PDF kuralı).

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/03_random_forest.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/03_random_forest.py --fast --sample-size 5000
"""

import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
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
    parser = argparse.ArgumentParser(description="Adım 6.4 Random Forest eğitimi")
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


# ─────────────────────────────────────────────
#  Feature Importance
# ─────────────────────────────────────────────

def extract_feature_importance(cv_model, feature_cols):
    """
    Best model'in featureImportances vektöründen en önemli 10 feature'ı çıkarır.
    Random Forest, ağaçlar arası ortalama Gini importance sağlar.

    Returns:
        list: [(feature_name, importance)] — en yüksekten düşüğe sıralı, top 10
    """
    best_pipeline_model = cv_model.bestModel
    rf_model = best_pipeline_model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    importance_pairs = [
        (feature_name, importance_value)
        for feature_name, importance_value in zip(feature_cols, importances)
    ]
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    return importance_pairs[:10]


def extract_all_feature_importance(cv_model, feature_cols):
    """
    Tüm feature'ların importance skorlarını döndürür (CSV kaydetmek için).

    Returns:
        list: [(feature_name, importance)] — tüm feature'lar, sıralı
    """
    best_pipeline_model = cv_model.bestModel
    rf_model = best_pipeline_model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    importance_pairs = [
        (feature_name, importance_value)
        for feature_name, importance_value in zip(feature_cols, importances)
    ]
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    return importance_pairs


# ─────────────────────────────────────────────
#  Cross Validation Sonuç Analizi
# ─────────────────────────────────────────────

def analyze_cv_results(cv_model, param_grid):
    """
    CrossValidator sonuçlarını analiz eder.
    Her parametre kombinasyonunun ortalama CV skorunu raporlar.

    Returns:
        dict: CV analiz bilgileri (best_score, all_scores, best_index)
    """
    avg_metrics = cv_model.avgMetrics
    best_index = int(max(range(len(avg_metrics)), key=lambda i: avg_metrics[i]))
    best_score = float(avg_metrics[best_index])

    print("\n📊 Cross Validation Sonuçları (AUC-ROC):")
    print(f"   {'#':<4} {'Ortalama AUC':<15} {'Parametreler'}")
    print(f"   {'─'*60}")

    cv_details = []
    for i, (params, score) in enumerate(zip(param_grid, avg_metrics)):
        param_str_parts = []
        for param, value in params.items():
            param_name = param.name
            param_str_parts.append(f"{param_name}={value}")
        param_str = ", ".join(param_str_parts)

        marker = " ← best" if i == best_index else ""
        print(f"   {i+1:<4} {score:<15.6f} {param_str}{marker}")

        cv_details.append({
            "index": i,
            "score": float(score),
            "params": param_str,
        })

    print(f"\n   ✅ En iyi CV skoru: {best_score:.6f} (kombinasyon #{best_index + 1})")

    return {
        "best_score": best_score,
        "best_index": best_index,
        "all_scores": [float(s) for s in avg_metrics],
        "details": cv_details,
    }


# ─────────────────────────────────────────────
#  Random Forest Ensemble Analizi
# ─────────────────────────────────────────────

def analyze_forest_structure(cv_model):
    """
    Eğitilmiş Random Forest'in yapısını analiz eder.

    Raporlananlar:
    - Ağaç sayısı (numTrees)
    - Her ağacın derinliği ve düğüm sayısı özeti
    - Toplam düğüm sayısı

    Returns:
        dict: Orman yapı bilgileri
    """
    best_pipeline_model = cv_model.bestModel
    rf_model = best_pipeline_model.stages[-1]

    num_trees = rf_model.getNumTrees
    total_nodes = rf_model.totalNumNodes

    # Her bir ağacın bilgisi
    tree_depths = []
    tree_nodes = []
    for tree in rf_model.trees:
        tree_depths.append(tree.depth)
        tree_nodes.append(tree.numNodes)

    avg_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0
    max_depth = max(tree_depths) if tree_depths else 0
    min_depth = min(tree_depths) if tree_depths else 0
    avg_nodes = sum(tree_nodes) / len(tree_nodes) if tree_nodes else 0

    print("\n🌲 Random Forest Yapı Analizi:")
    print(f"   Ağaç sayısı (numTrees):       {num_trees}")
    print(f"   Toplam düğüm sayısı:          {total_nodes}")
    print(f"   Ortalama ağaç derinliği:      {avg_depth:.1f}")
    print(f"   Min / Max ağaç derinliği:     {min_depth} / {max_depth}")
    print(f"   Ortalama düğüm/ağaç:          {avg_nodes:.1f}")

    # İlk ağacın debug string'inden kısa önizleme
    if rf_model.trees:
        first_tree_debug = rf_model.trees[0].toDebugString
        debug_lines = first_tree_debug.split("\n")
        print(f"\n   İlk ağaç önizleme (ilk 10 satır):")
        for line in debug_lines[:10]:
            print(f"   {line}")
        if len(debug_lines) > 10:
            print(f"   ... ({len(debug_lines) - 10} satır daha)")

    return {
        "num_trees": num_trees,
        "total_nodes": total_nodes,
        "avg_depth": round(avg_depth, 2),
        "max_depth": max_depth,
        "min_depth": min_depth,
        "avg_nodes_per_tree": round(avg_nodes, 2),
        "tree_depths": tree_depths,
        "tree_nodes": tree_nodes,
    }


# ─────────────────────────────────────────────
#  Feature Importance Bar Chart
# ─────────────────────────────────────────────

def save_feature_importance_chart(top_features, output_path):
    """
    En önemli feature'ları horizontal bar chart olarak kaydeder.

    Args:
        top_features: [(feature_name, importance)] listesi
        output_path: Grafik dosya yolu (.png)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        names = [f[0] for f in reversed(top_features)]
        values = [f[1] for f in reversed(top_features)]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            names, values,
            color="#4CAF50", edgecolor="#2E7D32", height=0.6,
        )

        # Değerleri bar'ların yanına yaz
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("Feature Importance (Mean Gini Impurity Decrease)", fontsize=11)
        ax.set_title(
            "Random Forest — Top 10 Feature Importance",
            fontsize=13, fontweight="bold",
        )
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


# ─────────────────────────────────────────────
#  Feature Importance CSV
# ─────────────────────────────────────────────

def save_feature_importance_csv(all_features, output_path):
    """
    Tüm feature importance skorlarını CSV dosyasına kaydeder.

    Args:
        all_features: [(feature_name, importance)] listesi (tüm feature'lar)
        output_path: CSV dosya yolu
    """
    try:
        csv_lines = ["rank,feature_name,importance"]
        for rank, (fname, importance) in enumerate(all_features, start=1):
            csv_lines.append(f"{rank},{fname},{importance:.8f}")

        with open(output_path, "w") as f:
            f.write("\n".join(csv_lines) + "\n")

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
    print("  Adım 6.4 — Random Forest Eğitimi")
    print("=" * 60)
    print(
        f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
        f"CV parallelism: {args.cv_parallelism}"
    )

    # ── 1) Spark başlat ──
    spark = get_spark("Model-3-RandomForest")

    # ── 2) Ortak pipeline (6.1 çıktıları) ──
    print("\n[1/8] Veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols = run_ml_pipeline(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    print(f"   ⏱️ Ortak pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature sayısı: {len(feature_cols)}")

    # ── 3) Random Forest + Pipeline ──
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

    # ── 4) Cross Validation — Hiperparametre Tune ──
    print("\n[3/8] Cross Validation ile hiperparametre optimizasyonu...")
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

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
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
    print(
        f"   📌 Grid boyutu: {len(param_grid)} kombinasyon | Fold: {num_folds} | "
        f"Toplam fit: {total_cv_runs}"
    )
    print(f"   📌 numTrees: {num_trees_values}")
    print(f"   📌 maxDepth: {max_depth_values}")
    print(f"   📌 minInstancesPerNode: {min_instances_values}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    cv_duration = time.perf_counter() - cv_start
    print(f"   ⏱️ Cross Validation süresi: {cv_duration:.2f}s")

    best_pipeline_model = cv_model.bestModel
    best_rf_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"numTrees={best_rf_model.getNumTrees}, "
        f"maxDepth={best_rf_model.getOrDefault('maxDepth')}, "
        f"minInstancesPerNode={best_rf_model.getOrDefault('minInstancesPerNode')}"
    )

    # ── 5) Cross Validation sonuçlarını analiz et ──
    print("\n[4/8] Cross Validation sonuçları analiz ediliyor...")
    cv_results = analyze_cv_results(cv_model, param_grid)

    # ── 6) Test seti değerlendirme ──
    print("\n[5/8] Test seti üzerinde değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model(predictions)
    confusion = compute_confusion_matrix(predictions)
    print(f"   ⏱️ Test değerlendirme süresi: {time.perf_counter() - eval_start:.2f}s")

    # ── 7) Random Forest yapı analizi ──
    print("\n[6/8] Random Forest yapısı analiz ediliyor...")
    forest_info = analyze_forest_structure(cv_model)

    # ── 8) Feature importance ──
    print("\n[7/8] Feature importance çıkarılıyor...")
    fi_start = time.perf_counter()
    top_features = extract_feature_importance(cv_model, feature_cols)
    all_features = extract_all_feature_importance(cv_model, feature_cols)

    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")

    # Horizontal bar chart kaydet
    chart_path = "/opt/bitnami/spark/ml/rf_feature_importance.png"
    save_feature_importance_chart(top_features, chart_path)

    # Feature importance CSV kaydet
    csv_path = "/opt/bitnami/spark/ml/rf_feature_importance.csv"
    save_feature_importance_csv(all_features, csv_path)

    print(f"   ⏱️ Feature importance süresi: {time.perf_counter() - fi_start:.2f}s")

    # ── 9) MLflow Logging ──
    print("\n[8/8] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()
    best_params = {
        "numTrees": int(best_rf_model.getNumTrees),
        "maxDepth": int(best_rf_model.getOrDefault("maxDepth")),
        "minInstancesPerNode": int(best_rf_model.getOrDefault("minInstancesPerNode")),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "forest_total_nodes": forest_info["total_nodes"],
        "forest_avg_depth": forest_info["avg_depth"],
        "forest_max_depth": forest_info["max_depth"],
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    run_name = "random_forest_v1_fast" if args.fast else "random_forest_v1"

    # Ek metrikler: CV best score ve orman yapı bilgileri
    metrics["cv_best_auc"] = cv_results["best_score"]
    metrics["forest_num_trees"] = float(forest_info["num_trees"])
    metrics["forest_total_nodes"] = float(forest_info["total_nodes"])
    metrics["forest_avg_depth"] = float(forest_info["avg_depth"])

    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="RandomForest",
        params=best_params,
        metrics=metrics,
        model=best_pipeline_model,
        confusion_matrix=confusion,
        feature_importance=top_features,
        tags={
            "task": "step_6_4",
            "model_index": "3",
            "classification_type": "binary",
            "ensemble_method": "bagging",
        },
    )

    # Ek artifact'leri MLflow'a logla
    try:
        import mlflow
        import os

        with mlflow.start_run(run_id=run_id):
            # CV sonuçlarını CSV olarak logla
            cv_csv_lines = ["combination,avg_auc_roc,params"]
            for detail in cv_results["details"]:
                cv_csv_lines.append(
                    f"{detail['index']+1},{detail['score']:.6f},{detail['params']}"
                )
            mlflow.log_text(
                "\n".join(cv_csv_lines) + "\n",
                "cv_results.csv",
            )

            # Feature importance CSV'sini artifact olarak ekle
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, "feature_importance")

            # Feature importance chart'ını artifact olarak ekle
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, "charts")

            # Orman yapı özetini text olarak logla
            forest_summary_lines = [
                "Random Forest Yapı Özeti",
                "=" * 40,
                f"Ağaç sayısı:              {forest_info['num_trees']}",
                f"Toplam düğüm:             {forest_info['total_nodes']}",
                f"Ortalama derinlik:        {forest_info['avg_depth']}",
                f"Min / Max derinlik:       {forest_info['min_depth']} / {forest_info['max_depth']}",
                f"Ortalama düğüm/ağaç:      {forest_info['avg_nodes_per_tree']}",
                "",
                "Ağaç Derinlikleri:",
            ]
            for i, (d, n) in enumerate(
                zip(forest_info["tree_depths"], forest_info["tree_nodes"])
            ):
                forest_summary_lines.append(f"  Ağaç {i+1:>3}: derinlik={d}, düğüm={n}")

            mlflow.log_text(
                "\n".join(forest_summary_lines) + "\n",
                "forest_structure.txt",
            )

        print("   ✅ CV sonuçları, chart ve orman yapısı MLflow'a loglandı.")
    except Exception as e:
        print(f"   ⚠️ Ek artifact'ler loglanamadı: {e}")

    print(f"   ⏱️ MLflow loglama süresi: {time.perf_counter() - mlflow_start:.2f}s")

    # ── Özet ──
    print("\n" + "=" * 60)
    print("  ✅ Adım 6.4 — Random Forest tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  MLflow UI: http://localhost:5000")
    print(f"  En iyi numTrees: {best_params['numTrees']}")
    print(f"  En iyi maxDepth: {best_params['maxDepth']}")
    print(f"  En iyi minInstancesPerNode: {best_params['minInstancesPerNode']}")
    print(f"  Orman — ağaç sayısı: {forest_info['num_trees']}")
    print(f"  Orman — toplam düğüm: {forest_info['total_nodes']}")
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
