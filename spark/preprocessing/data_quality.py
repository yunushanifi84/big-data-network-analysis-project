"""
Veri Kalite Analizi Modülü
==========================
Edge-IIoTset veri seti üzerinde kapsamlı veri kalite kontrolleri yapar.
Eksik değer, duplikat, outlier ve sınıf dağılımı raporları üretir.

Kullanım:
    from spark.preprocessing.data_quality import DataQualityAnalyzer
    analyzer = DataQualityAnalyzer(spark, df)
    report = analyzer.run_full_analysis()
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from typing import Dict, List, Any


class DataQualityAnalyzer:
    """Edge-IIoTset veri seti için kapsamlı veri kalite analiz aracı."""

    def __init__(self, spark: SparkSession, df: DataFrame):
        self.spark = spark
        self.df = df
        self._report: Dict[str, Any] = {}

    # ──────────────────────────────────────────────
    # 1. TEMEL İSTATİSTİKLER
    # ──────────────────────────────────────────────
    def basic_stats(self) -> Dict[str, Any]:
        """Satır sayısı, kolon sayısı ve veri tipleri."""
        total_rows = self.df.count()
        total_cols = len(self.df.columns)

        # Veri tipi dağılımı
        dtype_counts = {}
        for col_name, dtype in self.df.dtypes:
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        stats = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "dtype_distribution": dtype_counts,
        }
        self._report["basic_stats"] = stats
        print(f"📊 Temel İstatistikler:")
        print(f"   Satır sayısı : {total_rows:,}")
        print(f"   Kolon sayısı : {total_cols}")
        print(f"   Veri tipleri : {dtype_counts}")
        return stats

    # ──────────────────────────────────────────────
    # 2. EKSİK DEĞER ANALİZİ
    # ──────────────────────────────────────────────
    def missing_value_analysis(self) -> DataFrame:
        """Her kolon için eksik değer sayısı ve yüzdesi."""
        total_rows = self.df.count()

        # Null + NaN + sıfır uzunluklu string kontrolleri
        missing_exprs = []
        for col_name in self.df.columns:
            missing_exprs.append(
                F.sum(
                    F.when(
                        F.col(f"`{col_name}`").isNull() |
                        F.isnan(F.col(f"`{col_name}`").cast("double")) |
                        (F.col(f"`{col_name}`") == ""),
                        1
                    ).otherwise(0)
                ).alias(col_name)
            )

        missing_counts = self.df.select(missing_exprs).collect()[0]

        # DataFrame olarak döndür
        rows = []
        for col_name in self.df.columns:
            count = missing_counts[col_name]
            if count is None:
                count = 0
            pct = round((count / total_rows) * 100, 2) if total_rows > 0 else 0
            rows.append((col_name, int(count), pct))

        missing_df = self.spark.createDataFrame(
            rows, ["column_name", "missing_count", "missing_pct"]
        ).orderBy(F.desc("missing_count"))

        self._report["missing_values"] = rows

        # Problemli kolonları raporla
        problem_cols = [r for r in rows if r[1] > 0]
        print(f"\n🔍 Eksik Değer Analizi:")
        print(f"   Toplam kolon      : {len(self.df.columns)}")
        print(f"   Eksik değerli     : {len(problem_cols)}")
        if problem_cols:
            print(f"   En kötü 5 kolon   :")
            for col, cnt, pct in sorted(problem_cols, key=lambda x: -x[1])[:5]:
                print(f"     - {col}: {cnt:,} ({pct}%)")

        return missing_df

    # ──────────────────────────────────────────────
    # 3. DUPLİKAT ANALİZİ
    # ──────────────────────────────────────────────
    def duplicate_analysis(self) -> Dict[str, int]:
        """Tekrarlayan satırların tespiti."""
        total = self.df.count()
        distinct = self.df.distinct().count()
        duplicates = total - distinct

        result = {
            "total_rows": total,
            "distinct_rows": distinct,
            "duplicate_rows": duplicates,
            "duplicate_pct": round((duplicates / total) * 100, 2) if total > 0 else 0,
        }
        self._report["duplicates"] = result

        print(f"\n📋 Duplikat Analizi:")
        print(f"   Toplam satır  : {total:,}")
        print(f"   Benzersiz     : {distinct:,}")
        print(f"   Duplikat      : {duplicates:,} ({result['duplicate_pct']}%)")

        return result

    # ──────────────────────────────────────────────
    # 4. SINIF DAĞILIMI
    # ──────────────────────────────────────────────
    def class_distribution(self) -> DataFrame:
        """Attack_label ve Attack_type sınıf dağılımı."""
        total = self.df.count()

        # Binary dağılım (Normal vs Attack)
        binary_dist = (
            self.df.groupBy("Attack_label")
            .count()
            .withColumn("percentage", F.round(F.col("count") / total * 100, 2))
            .orderBy("Attack_label")
        )

        print(f"\n⚔️ Sınıf Dağılımı (Binary):")
        binary_dist.show(truncate=False)

        # Saldırı tipi dağılımı
        attack_dist = (
            self.df.groupBy("Attack_type")
            .count()
            .withColumn("percentage", F.round(F.col("count") / total * 100, 2))
            .orderBy(F.desc("count"))
        )

        print(f"⚔️ Saldırı Tipi Dağılımı:")
        attack_dist.show(20, truncate=False)

        self._report["class_distribution"] = {
            "binary": binary_dist.collect(),
            "multiclass": attack_dist.collect(),
        }

        return attack_dist

    # ──────────────────────────────────────────────
    # 5. SAYISAL KOLON ANALİZİ
    # ──────────────────────────────────────────────
    def numeric_column_stats(self) -> DataFrame:
        """Sayısal kolonların min, max, mean, std değerleri."""
        numeric_cols = [
            col_name for col_name, dtype in self.df.dtypes
            if dtype in ("double", "float", "int", "bigint", "long")
        ]

        if not numeric_cols:
            print("⚠️ Sayısal kolon bulunamadı.")
            return self.spark.createDataFrame([], "column_name STRING")

        # PySpark'ın describe() fonksiyonu noktalı kolon isimlerinde hata verdiği için 
        # geçici olarak noktaları alt çizgi (_) ile değiştiriyoruz.
        temp_df = self.df.select([F.col(f"`{c}`").alias(c.replace(".", "_")) for c in numeric_cols])
        stats_df = temp_df.describe()

        print(f"\n📈 Sayısal Kolon İstatistikleri:")
        print(f"   Sayısal kolon sayısı: {len(numeric_cols)}")

        return stats_df

    # ──────────────────────────────────────────────
    # 6. INF DEĞER KONTROLÜ
    # ──────────────────────────────────────────────
    def infinity_check(self) -> Dict[str, int]:
        """Sonsuz (inf/-inf) değer tespiti."""
        numeric_cols = [
            col_name for col_name, dtype in self.df.dtypes
            if dtype in ("double", "float")
        ]

        inf_counts = {}
        for col_name in numeric_cols:
            count = self.df.filter(
                F.col(f"`{col_name}`").isin(float("inf"), float("-inf"))
            ).count()
            if count > 0:
                inf_counts[col_name] = count

        self._report["infinity_values"] = inf_counts

        print(f"\n♾️ Sonsuz Değer (Inf) Kontrolü:")
        if inf_counts:
            for col, cnt in sorted(inf_counts.items(), key=lambda x: -x[1]):
                print(f"   - {col}: {cnt:,} inf değer")
        else:
            print("   ✅ Sonsuz değer bulunamadı.")

        return inf_counts

    # ──────────────────────────────────────────────
    # TOPLU ANALİZ
    # ──────────────────────────────────────────────
    def run_full_analysis(self) -> Dict[str, Any]:
        """Tüm veri kalite analizlerini sırayla çalıştırır."""
        print("=" * 60)
        print("  VERİ KALİTE ANALİZİ — Edge-IIoTset")
        print("=" * 60)

        self.basic_stats()
        self.missing_value_analysis()
        self.duplicate_analysis()
        self.class_distribution()
        self.numeric_column_stats()
        self.infinity_check()

        print("\n" + "=" * 60)
        print("  ✅ Veri kalite analizi tamamlandı.")
        print("=" * 60)

        return self._report
