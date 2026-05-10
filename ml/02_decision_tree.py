"""
Adım 6.3 — Model 2: Decision Tree
====================================
Yorumlanabilir bir binary sınıflandırma modeli olarak Decision Tree eğitir,
hiperparametre optimizasyonu uygular ve sonuçları MLflow'a loglar.

Avantajlar:
  - Yorumlanabilirlik: karar ağacı yapısı (derinlik, düğüm sayısı) kolayca raporlanır.
  - Feature importance doğrudan modelden çıkarılabilir.

Riskler:
  - Overfitting: maxDepth sınırlı tutulmalıdır.

Çalıştırma:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/02_decision_tree.py

Hızlı test:
    docker exec spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/02_decision_tree.py --fast --sample-size 5000
"""

import argparse
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
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
    parser = argparse.ArgumentParser(description="Adım 6.3 Decision Tree eğitimi")
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
    Decision Tree, Gini/Entropy tabanlı feature importance sağlar.
    """
    best_pipeline_model = cv_model.bestModel
    dt_model = best_pipeline_model.stages[-1]
    importances = dt_model.featureImportances.toArray().tolist()

    importance_pairs = [
        (feature_name, importance_value)
        for feature_name, importance_value in zip(feature_cols, importances)
    ]
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    return importance_pairs[:10]


# ─────────────────────────────────────────────
#  Karar Ağacı Yapı Analizi
# ─────────────────────────────────────────────

def analyze_tree_structure(cv_model):
    """
    Eğitilmiş karar ağacının yapısını analiz eder ve raporlar.

    Raporlananlar:
    - Ağaç derinliği (depth)
    - Toplam düğüm sayısı (numNodes)
    - Karar kuralı debug string'i (toDebugString özeti)

    Returns:
        dict: Ağaç yapı bilgileri
    """
    best_pipeline_model = cv_model.bestModel
    dt_model = best_pipeline_model.stages[-1]

    depth = dt_model.depth
    num_nodes = dt_model.numNodes

    debug_string = dt_model.toDebugString
    # Debug string çok uzun olabilir — ilk 50 satırı al
    debug_lines = debug_string.split("\n")
    debug_preview = "\n".join(debug_lines[:50])
    if len(debug_lines) > 50:
        debug_preview += f"\n... ({len(debug_lines) - 50} satır daha)"

    print("\n🌳 Karar Ağacı Yapı Analizi:")
    print(f"   Derinlik (depth):      {depth}")
    print(f"   Toplam düğüm sayısı:   {num_nodes}")
    print(f"   Yaprak düğüm sayısı:   ~{(num_nodes + 1) // 2}")
    print(f"   Debug string satır:    {len(debug_lines)}")
    print(f"\n   Ağaç yapısı önizleme (ilk 20 satır):")
    for line in debug_lines[:20]:
        print(f"   {line}")
    if len(debug_lines) > 20:
        print(f"   ... ({len(debug_lines) - 20} satır daha)")

    return {
        "depth": depth,
        "num_nodes": num_nodes,
        "num_leaves": (num_nodes + 1) // 2,
        "debug_string_lines": len(debug_lines),
        "debug_string_preview": debug_preview,
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
        bars = ax.barh(names, values, color="#2196F3", edgecolor="#1565C0", height=0.6)

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

        ax.set_xlabel("Feature Importance (Gini/Entropy)", fontsize=11)
        ax.set_title("Decision Tree — Top 10 Feature Importance", fontsize=13, fontweight="bold")
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
#  Ana Akış
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 60)
    print("  Adım 6.3 — Decision Tree Eğitimi")
    print("=" * 60)
    print(
        f"  Mod: {'fast' if args.fast else 'normal'} | CV: {args.cv_mode} | "
        f"CV parallelism: {args.cv_parallelism}"
    )

    # ── 1) Spark başlat ──
    spark = get_spark("Model-2-DecisionTree")

    # ── 2) Ortak pipeline (6.1 çıktıları) ──
    print("\n[1/7] Veri pipeline hazırlanıyor...")
    stage_start = time.perf_counter()
    train_df, test_df, feature_cols = run_ml_pipeline(
        spark,
        sample_size=args.sample_size if args.fast else None,
        split_log_stats=not args.fast,
    )
    print(f"   ⏱️ Ortak pipeline süresi: {time.perf_counter() - stage_start:.2f}s")
    print(f"   📌 Feature sayısı: {len(feature_cols)}")

    # ── 3) Decision Tree + Pipeline ──
    print("\n[2/7] Decision Tree pipeline kuruluyor...")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxDepth=10,
        seed=42,
    )
    pipeline = Pipeline(stages=[dt])

    # ── 4) Cross Validation — Hiperparametre Tune ──
    print("\n[3/7] Cross Validation ile hiperparametre optimizasyonu...")
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
    print(f"   📌 maxDepth: {max_depth_values}")
    print(f"   📌 minInstancesPerNode: {min_instances_values}")
    print(f"   📌 impurity: {impurity_values}")

    cv_start = time.perf_counter()
    cv_model = cv.fit(train_df)
    print(f"   ⏱️ Cross Validation süresi: {time.perf_counter() - cv_start:.2f}s")

    best_pipeline_model = cv_model.bestModel
    best_dt_model = best_pipeline_model.stages[-1]
    print(
        "   ✅ En iyi parametreler: "
        f"maxDepth={best_dt_model.getOrDefault('maxDepth')}, "
        f"minInstancesPerNode={best_dt_model.getOrDefault('minInstancesPerNode')}, "
        f"impurity={best_dt_model.getOrDefault('impurity')}"
    )

    # ── 5) Test seti değerlendirme ──
    print("\n[4/7] Test seti üzerinde değerlendirme...")
    eval_start = time.perf_counter()
    predictions = best_pipeline_model.transform(test_df)
    metrics = evaluate_model(predictions)
    confusion = compute_confusion_matrix(predictions)
    print(f"   ⏱️ Test değerlendirme süresi: {time.perf_counter() - eval_start:.2f}s")

    # ── 6) Karar ağacı yapı analizi ──
    print("\n[5/7] Karar ağacı yapısı yorumlanıyor...")
    tree_info = analyze_tree_structure(cv_model)

    # ── 7) Feature importance ──
    print("\n[6/7] Feature importance çıkarılıyor...")
    fi_start = time.perf_counter()
    top_features = extract_feature_importance(cv_model, feature_cols)
    print("   Top 10 feature:")
    for idx, (fname, importance) in enumerate(top_features, start=1):
        print(f"   {idx:>2}. {fname:<30} {importance:.6f}")

    # Horizontal bar chart kaydet
    chart_path = "/opt/bitnami/spark/ml/dt_feature_importance.png"
    save_feature_importance_chart(top_features, chart_path)
    print(f"   ⏱️ Feature importance süresi: {time.perf_counter() - fi_start:.2f}s")

    # ── 8) MLflow Logging ──
    print("\n[7/7] MLflow'a loglanıyor...")
    mlflow_start = time.perf_counter()
    best_params = {
        "maxDepth": int(best_dt_model.getOrDefault("maxDepth")),
        "minInstancesPerNode": int(best_dt_model.getOrDefault("minInstancesPerNode")),
        "impurity": str(best_dt_model.getOrDefault("impurity")),
        "numFolds": num_folds,
        "grid_size": len(param_grid),
        "cv_total_fits": total_cv_runs,
        "cv_mode": args.cv_mode,
        "cv_parallelism": int(args.cv_parallelism),
        "weightCol": "classWeight",
        "tree_depth": tree_info["depth"],
        "tree_num_nodes": tree_info["num_nodes"],
        "tree_num_leaves": tree_info["num_leaves"],
    }
    if args.fast:
        best_params["fast_mode"] = True
        best_params["sample_size"] = int(args.sample_size)

    run_name = "decision_tree_v1_fast" if args.fast else "decision_tree_v1"

    # Ağaç yapısı metrikleri ek olarak logla
    metrics["tree_depth"] = float(tree_info["depth"])
    metrics["tree_num_nodes"] = float(tree_info["num_nodes"])

    run_id = log_to_mlflow(
        run_name=run_name,
        model_type="DecisionTree",
        params=best_params,
        metrics=metrics,
        model=best_pipeline_model,
        confusion_matrix=confusion,
        feature_importance=top_features,
        tags={
            "task": "step_6_3",
            "model_index": "2",
            "classification_type": "binary",
            "interpretable": "true",
        },
    )

    # Ağaç yapısı debug string'ini MLflow'a artifact olarak logla
    try:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_text(
                tree_info["debug_string_preview"],
                "tree_structure.txt",
            )
            # Feature importance chart'ı da artifact olarak ekle
            try:
                import os
                if os.path.exists(chart_path):
                    mlflow.log_artifact(chart_path, "charts")
            except Exception as e:
                print(f"   ⚠️ Chart artifact loglanamadı: {e}")

        print("   ✅ Ağaç yapısı ve chart MLflow'a loglandı.")
    except Exception as e:
        print(f"   ⚠️ Ağaç yapısı loglanamadı: {e}")

    print(f"   ⏱️ MLflow loglama süresi: {time.perf_counter() - mlflow_start:.2f}s")

    # ── Özet ──
    print("\n" + "=" * 60)
    print("  ✅ Adım 6.3 — Decision Tree tamamlandı!")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  MLflow UI: http://localhost:5000")
    print(f"  En iyi maxDepth: {best_params['maxDepth']}")
    print(f"  En iyi impurity: {best_params['impurity']}")
    print(f"  Ağaç derinliği: {tree_info['depth']}")
    print(f"  Düğüm sayısı: {tree_info['num_nodes']}")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    print(f"  Toplam eğitim süresi: {time.perf_counter() - total_start:.2f}s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
