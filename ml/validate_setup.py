"""
Adım 6.1 — MLflow Altyapısı Doğrulama Scripti
===============================================
MLflow bağlantısını test eder, Gold Delta verisini yükler,
feature hazırlığını yapar ve train/test split doğrulaması gerçekleştirir.

Çalıştırma:
    docker exec -it spark-master spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/bitnami/spark/ml/validate_setup.py
"""

import sys
import argparse

# Proje dizinini Python path'e ekle
sys.path.insert(0, "/opt/bitnami/spark")

from spark.spark_session import get_spark
from ml.utils import (
    init_mlflow,
    load_gold_data,
    get_feature_columns,
    prepare_features,
    split_data,
    compute_class_weights,
    add_weight_column,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Adım 6.1 doğrulama scripti")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Hızlı doğrulama modu (örneklem veri ile çalışır)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Hızlı modda kullanılacak maksimum satır sayısı",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Adım 6.1 — MLflow Altyapısı Doğrulama")
    print("=" * 60)

    # 1. Spark Session
    print("\n[1/6] SparkSession başlatılıyor...")
    spark = get_spark("MLflow-Validation")

    # 2. MLflow bağlantısı
    print("\n[2/6] MLflow bağlantısı test ediliyor...")
    try:
        init_mlflow()
        import mlflow
        # Test run başlat ve kapat
        with mlflow.start_run(run_name="validation_test") as run:
            mlflow.log_param("test", "connection_check")
            mlflow.log_metric("validation_status", 1.0)
            mlflow.set_tag("stage", "validation")
        print("   ✅ MLflow test run başarılı!")
    except Exception as e:
        print(f"   ❌ MLflow bağlantı hatası: {e}")
        print("   MLflow server çalışıyor mu kontrol edin: docker logs mlflow-server")
        spark.stop()
        sys.exit(1)

    # 3. Gold katmanından veri yükle
    print("\n[3/6] Gold katmanından veri yükleniyor...")
    try:
        df = load_gold_data(spark)
        if args.fast:
            print(f"\n   ⚡ Hızlı mod aktif: veri {args.sample_size:,} satır ile sınırlandırılıyor.")
            df = df.limit(args.sample_size)
            df = df.cache()
            _ = df.count()  # cache materialize
        print("\n   İlk 5 satır:")
        df.show(5, truncate=True)
        print(f"\n   Şema:")
        df.printSchema()
    except Exception as e:
        print(f"   ❌ Gold veri yükleme hatası: {e}")
        spark.stop()
        sys.exit(1)

    # 4. Feature hazırlığı
    print("\n[4/6] Feature hazırlığı yapılıyor...")
    try:
        feature_cols = get_feature_columns(df)
        print(f"   Feature kolonları ({len(feature_cols)} adet):")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"   {i:>3}. {col}")
        if len(feature_cols) > 10:
            print(f"   ... ve {len(feature_cols) - 10} kolon daha.")

        prepared_df = prepare_features(df, feature_cols=feature_cols)
    except Exception as e:
        print(f"   ❌ Feature hazırlığı hatası: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)

    # 5. Sınıf dağılımı ve ağırlıklar
    print("\n[5/6] Sınıf ağırlıkları hesaplanıyor...")
    try:
        weights = compute_class_weights(prepared_df)
        prepared_df = add_weight_column(prepared_df, weights)
    except Exception as e:
        print(f"   ❌ Sınıf ağırlığı hatası: {e}")

    # 6. Train/Test split
    print("\n[6/6] Train/Test split yapılıyor...")
    try:
        train_df, test_df = split_data(prepared_df, log_stats=not args.fast)
        if not args.fast:
            print(f"\n   Train features örneği:")
            train_df.select("features", "label", "classWeight").show(3, truncate=True)
    except Exception as e:
        print(f"   ❌ Split hatası: {e}")

    # Özet
    print("\n" + "=" * 60)
    print("  ✅ Adım 6.1 DOĞRULAMA BAŞARILI!")
    print("=" * 60)
    print(f"\n  MLflow UI:  http://localhost:5000")
    print(f"  Gold Veri:  delta-storage/gold/ml_ready/")
    print(f"  Feature #:  {len(feature_cols)}")
    print(f"\n  Bir sonraki adım: Adım 6.2 — Logistic Regression")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
