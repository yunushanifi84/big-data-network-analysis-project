"""
Adım 5 — Veri Ön İşleme ve Feature Engineering Pipeline
========================================================
Bu script Edge-IIoTset veri setini yükler, kalite analizini yapar,
temizler, 5 yeni feature üretir ve sonuçları Delta Lake'e kaydeder.

Çalıştırma (Docker konteyner içinde):
    spark-submit --packages io.delta:delta-core_2.12:2.4.0 \
        spark/preprocessing/run_pipeline.py

Veya JupyterLab üzerinden:
    %run spark/preprocessing/run_pipeline.py
"""

import sys
import os
import time

# Proje kök dizinini Python path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spark.spark_session import get_spark
from spark.preprocessing.data_quality import DataQualityAnalyzer
from spark.preprocessing.data_cleaner import DataCleaner
from spark.preprocessing.feature_engineering import FeatureEngineer


# ══════════════════════════════════════════════════
# KONFİGÜRASYON
# ══════════════════════════════════════════════════
RAW_CSV_PATH = os.path.join(
    project_root,
    "data", "raw", "Edge-IIoTset dataset",
    "Selected dataset for ML and DL",
    "ML-EdgeIIoT-dataset.csv"
)
DELTA_SILVER_PATH = os.path.join(project_root, "delta-storage", "silver", "network_traffic")
DELTA_GOLD_PATH = os.path.join(project_root, "delta-storage", "gold", "ml_ready")


def main():
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  ADIM 5 — Veri Ön İşleme ve Feature Engineering        ║")
    print("║  IoT Network Intrusion Detection Pipeline               ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. SPARK SESSION ──
    print("\n📡 SparkSession başlatılıyor...")
    spark = get_spark("Step5-Preprocessing")
    print("   ✅ SparkSession hazır.")

    # ── 2. VERİ YÜKLEME ──
    print(f"\n📂 Veri seti yükleniyor...")
    print(f"   Dosya: {RAW_CSV_PATH}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(RAW_CSV_PATH)
    )

    print(f"   ✅ Yüklendi: {df.count():,} satır × {len(df.columns)} kolon")

    # ── 3. VERİ KALİTE ANALİZİ ──
    analyzer = DataQualityAnalyzer(spark, df)
    quality_report = analyzer.run_full_analysis()

    # ── 4. VERİ TEMİZLEME ──
    cleaner = DataCleaner(spark, df)
    clean_df = cleaner.clean()

    # ── 5. SILVER LAYER'A KAYDET ──
    print(f"\n💾 Silver Layer'a kaydediliyor...")
    print(f"   Path: {DELTA_SILVER_PATH}")
    (
        clean_df.write
        .format("delta")
        .mode("overwrite")
        .save(DELTA_SILVER_PATH)
    )
    silver_count = spark.read.format("delta").load(DELTA_SILVER_PATH).count()
    print(f"   ✅ Silver Layer: {silver_count:,} satır kaydedildi.")

    # ── 6. FEATURE ENGINEERING ──
    engineer = FeatureEngineer(spark, clean_df)
    featured_df = engineer.create_all_features()

    # Feature istatistikleri
    print("\n📊 Feature İstatistikleri:")
    engineer.get_feature_summary().show()

    # ── 7. GOLD LAYER'A KAYDET ──
    print(f"\n💾 Gold Layer'a kaydediliyor (ML-ready)...")
    print(f"   Path: {DELTA_GOLD_PATH}")
    (
        featured_df.write
        .format("delta")
        .mode("overwrite")
        .save(DELTA_GOLD_PATH)
    )
    gold_count = spark.read.format("delta").load(DELTA_GOLD_PATH).count()
    print(f"   ✅ Gold Layer: {gold_count:,} satır kaydedildi.")

    # ── 8. DOĞRULAMA ──
    print("\n" + "=" * 60)
    print("  DOĞRULAMA")
    print("=" * 60)

    gold_df = spark.read.format("delta").load(DELTA_GOLD_PATH)
    feature_names = engineer.get_feature_names()

    print(f"\n   📋 Gold Layer Kolon Sayısı: {len(gold_df.columns)}")
    print(f"   📋 Üretilen Feature'lar ({len(feature_names)}):")
    for name in feature_names:
        sample_values = gold_df.select(name).summary("min", "max", "mean").collect()
        print(f"      ✅ {name} — mevcut")

    # Feature'ların null olmadığını doğrula
    for name in feature_names:
        null_count = gold_df.filter(gold_df[name].isNull()).count()
        assert null_count == 0, f"❌ {name} kolonunda {null_count} null değer var!"

    print(f"\n   ✅ Tüm feature'lar null-free doğrulandı.")

    # ── ÖZET ──
    elapsed = round(time.time() - start_time, 1)
    print("\n╔" + "═" * 58 + "╗")
    print("║  ✅ ADIM 5 TAMAMLANDI                                   ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Ham veri    : {df.count():>10,} satır                       ║")
    print(f"║  Silver      : {silver_count:>10,} satır (temizlenmiş)        ║")
    print(f"║  Gold        : {gold_count:>10,} satır (feature eklenmiş)    ║")
    print(f"║  Feature     : {len(feature_names):>10} adet                          ║")
    print(f"║  Süre        : {elapsed:>10} saniye                       ║")
    print("╚" + "═" * 58 + "╝")

    spark.stop()


if __name__ == "__main__":
    main()
