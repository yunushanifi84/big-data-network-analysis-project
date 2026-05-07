"""
Veri Temizleme Modülü
=====================
Edge-IIoTset veri setine özel temizleme pipeline'ı.
Eksik değer doldurma, inf temizleme, tip dönüşümü ve duplikat kaldırma.

Kullanım:
    from spark.preprocessing.data_cleaner import DataCleaner
    cleaner = DataCleaner(spark, df)
    clean_df = cleaner.clean()
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List, Optional


class DataCleaner:
    """Edge-IIoTset veri seti için veri temizleme pipeline'ı."""

    # Analizde kullanılmayacak gereksiz kolonlar
    DROP_COLUMNS = [
        "arp.dst.proto_ipv4",   # Çoğunlukla 0
        "arp.opcode",           # Çoğunlukla 0
        "arp.hw.size",          # Çoğunlukla 0
        "arp.src.proto_ipv4",   # Çoğunlukla 0
        "icmp.checksum",        # Çoğunlukla 0
        "icmp.seq_le",          # Çoğunlukla 0
        "icmp.transmit_timestamp",  # Çoğunlukla 0
        "icmp.unused",          # Çoğunlukla 0
    ]

    def __init__(self, spark: SparkSession, df: DataFrame):
        self.spark = spark
        self.df = df
        self._log: List[str] = []

    def _log_step(self, message: str, before: int, after: int):
        """Temizleme adımını logla."""
        removed = before - after
        self._log.append(f"{message}: {removed:,} satır silindi")
        print(f"   🧹 {message}: {before:,} → {after:,} ({removed:,} silindi)")

    # ──────────────────────────────────────────────
    # 1. GEREKSIZ KOLONLARI KALDIR
    # ──────────────────────────────────────────────
    def drop_unnecessary_columns(self, extra_cols: Optional[List[str]] = None) -> "DataCleaner":
        """Analiz için gereksiz kolonları kaldırır."""
        cols_to_drop = self.DROP_COLUMNS.copy()
        if extra_cols:
            cols_to_drop.extend(extra_cols)

        # Sadece var olan kolonları drop et
        existing = [c for c in cols_to_drop if c in self.df.columns]
        self.df = self.df.drop(*existing)

        print(f"   🗑️ {len(existing)} gereksiz kolon kaldırıldı")
        self._log.append(f"Gereksiz kolonlar kaldırıldı: {existing}")
        return self

    # ──────────────────────────────────────────────
    # 2. INF DEĞERLERİ TEMİZLE
    # ──────────────────────────────────────────────
    def replace_infinity(self) -> "DataCleaner":
        """Sonsuz (inf/-inf) değerleri null'a çevir, sonra medyan ile doldur."""
        numeric_cols = [
            col_name for col_name, dtype in self.df.dtypes
            if dtype in ("double", "float")
        ]

        before = self.df.count()
        for col_name in numeric_cols:
            self.df = self.df.withColumn(
                col_name,
                F.when(
                    F.col(f"`{col_name}`").isin(float("inf"), float("-inf")),
                    None
                ).otherwise(F.col(f"`{col_name}`"))
            )

        print(f"   ♾️ Inf değerler null'a çevrildi ({len(numeric_cols)} sayısal kolon)")
        self._log.append("Inf değerler null'a çevrildi")
        return self

    # ──────────────────────────────────────────────
    # 3. EKSİK DEĞERLERİ DOLDUR
    # ──────────────────────────────────────────────
    def fill_missing_values(self) -> "DataCleaner":
        """
        Eksik değerleri doldur:
        - Sayısal kolonlar: 0 ile doldur (ağ trafiğinde 0 = aktivite yok)
        - String kolonlar: "unknown" ile doldur
        """
        numeric_cols = [
            col_name for col_name, dtype in self.df.dtypes
            if dtype in ("double", "float", "int", "bigint", "long")
        ]
        string_cols = [
            col_name for col_name, dtype in self.df.dtypes
            if dtype == "string"
        ]

        # Sayısal kolonları 0 ile doldur
        fill_values = {f"`{col}`": 0.0 for col in numeric_cols}
        fill_values.update({f"`{col}`": "unknown" for col in string_cols})

        self.df = self.df.fillna(fill_values)

        print(f"   📝 Eksik değerler dolduruldu (sayısal: 0, string: 'unknown')")
        self._log.append("Eksik değerler dolduruldu")
        return self

    # ──────────────────────────────────────────────
    # 4. VERİ TİPİ DÖNÜŞÜMLERİ
    # ──────────────────────────────────────────────
    def cast_types(self) -> "DataCleaner":
        """Veri tiplerini doğru formata dönüştür."""
        # frame.time → double olarak kalabilir (epoch timestamp)
        # Attack_label → integer (0 veya 1)
        if "Attack_label" in self.df.columns:
            self.df = self.df.withColumn(
                "Attack_label", F.col("Attack_label").cast("integer")
            )

        # tcp/udp port'larını integer'a çevir
        port_cols = [c for c in self.df.columns if "port" in c.lower()]
        for col_name in port_cols:
            self.df = self.df.withColumn(
                col_name, F.col(f"`{col_name}`").cast("integer")
            )

        print(f"   🔄 Veri tipleri düzeltildi (Attack_label→int, port kolonları→int)")
        self._log.append("Veri tipleri düzeltildi")
        return self

    # ──────────────────────────────────────────────
    # 5. DUPLİKATLARI KALDIR
    # ──────────────────────────────────────────────
    def remove_duplicates(self) -> "DataCleaner":
        """Tam duplikat satırları kaldır."""
        before = self.df.count()
        self.df = self.df.dropDuplicates()
        after = self.df.count()
        self._log_step("Duplikatlar kaldırıldı", before, after)
        return self

    # ──────────────────────────────────────────────
    # TAM TEMİZLEME PIPELINE
    # ──────────────────────────────────────────────
    def clean(self) -> DataFrame:
        """Tüm temizleme adımlarını sırayla çalıştırır."""
        print("\n" + "=" * 60)
        print("  VERİ TEMİZLEME PIPELINE'I")
        print("=" * 60)

        initial_count = self.df.count()
        initial_cols = len(self.df.columns)

        (
            self
            .drop_unnecessary_columns()
            .replace_infinity()
            .fill_missing_values()
            .cast_types()
            .remove_duplicates()
        )

        final_count = self.df.count()
        final_cols = len(self.df.columns)

        print(f"\n   ✅ Temizleme tamamlandı:")
        print(f"      Satır: {initial_count:,} → {final_count:,}")
        print(f"      Kolon: {initial_cols} → {final_cols}")
        print("=" * 60)

        return self.df

    def get_log(self) -> List[str]:
        """Temizleme adımlarının logunu döndürür."""
        return self._log
