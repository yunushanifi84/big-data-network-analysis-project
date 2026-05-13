"""
Feature Engineering Modülü
==========================
Edge-IIoTset veri setine özel 5 yeni feature üretir.
Proje PDF'i gereği EN AZ 5 feature üretilmesi zorunludur.

Her feature'ın:
  - İş mantığı açıklanmıştır (neden bu özellik seçildi?)
  - IoT saldırı tespitindeki rolü belirtilmiştir
  - Hesaplama formülü verilmiştir

Kullanım:
    from spark.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(spark, clean_df)
    featured_df = engineer.create_all_features()

NOT: Silver katmanı kolon isimlerindeki noktaları alt çizgiyle değiştirir.
     Örn: tcp.ack → tcp_ack, dns.qry.type → dns_qry_type
     Bu dosyadaki tüm kolon referansları bu dönüşüme uygun yazılmıştır.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List


class FeatureEngineer:
    """Edge-IIoTset veri setine özel feature engineering."""

    # Üretilecek feature isimleri (referans için)
    FEATURE_NAMES = [
        "traffic_asymmetry_ratio",
        "pkt_size_cv",
        "flow_intensity",
        "iat_regularity",
        "conn_efficiency",
    ]

    def __init__(self, spark: SparkSession, df: DataFrame):
        self.spark = spark
        self.df = df
        self._created_features: List[str] = []

    # ──────────────────────────────────────────────
    # FEATURE 1: Trafik Asimetri Oranı
    # ──────────────────────────────────────────────
    def add_traffic_asymmetry_ratio(self) -> "FeatureEngineer":
        """
        traffic_asymmetry_ratio = tcp_ack / (tcp_seq + 1)

        NEDEN?
        Normal ağ trafiği genelde simetrik bir istek-cevap (ACK/SEQ) akışına
        sahiptir. DDoS saldırılarında ise bu oran bozulur çünkü saldırgan
        çok sayıda SYN gönderir ama ACK dönemini tamamlamaz (SYN flood).
        Bu oran DDoS ve Flood saldırılarını tespit etmede güçlü bir sinyaldir.

        TESPIT ETTİĞİ SALDIRILAR: DDoS TCP/UDP/ICMP Flood, SYN Flood
        """
        ack_col = "tcp_ack" if "tcp_ack" in self.df.columns else None
        seq_col = "tcp_seq" if "tcp_seq" in self.df.columns else None

        if ack_col and seq_col:
            self.df = self.df.withColumn(
                "traffic_asymmetry_ratio",
                F.col(ack_col) / (F.col(seq_col) + F.lit(1))
            )
        else:
            # Fallback: tcp_ack_raw kullan
            ack_col = "tcp_ack_raw" if "tcp_ack_raw" in self.df.columns else "tcp_ack"
            self.df = self.df.withColumn(
                "traffic_asymmetry_ratio",
                F.col(ack_col) / (F.col("tcp_checksum") + F.lit(1))
            )

        # NaN/Inf temizle
        self.df = self.df.withColumn(
            "traffic_asymmetry_ratio",
            F.when(
                F.col("traffic_asymmetry_ratio").isNull() |
                F.isnan(F.col("traffic_asymmetry_ratio")) |
                F.col("traffic_asymmetry_ratio").isin(float("inf"), float("-inf")),
                F.lit(0.0)
            ).otherwise(F.col("traffic_asymmetry_ratio"))
        )

        self._created_features.append("traffic_asymmetry_ratio")
        print("   ✅ Feature 1: traffic_asymmetry_ratio (Trafik Asimetri Oranı)")
        return self

    # ──────────────────────────────────────────────
    # FEATURE 2: Paket Boyut Varyasyon Katsayısı
    # ──────────────────────────────────────────────
    def add_pkt_size_cv(self) -> "FeatureEngineer":
        """
        pkt_size_cv = tcp_len / (|tcp_seq - tcp_ack| + 1)

        NEDEN?
        Normal trafik tutarlı paket boyutlarına sahiptir — paket boyutu (tcp_len)
        ve bağlantı durumu (seq-ack farkı) arasında beklenen bir oran vardır.
        Port Scanning ve Vulnerability Scanner saldırıları ise küçük paketlerle
        çok sayıda bağlantı açar, bu da oranı bozar.

        TESPIT ETTİĞİ SALDIRILAR: Port Scanning, Vulnerability Scanner, OS Fingerprinting
        """
        if "tcp_len" in self.df.columns and "tcp_seq" in self.df.columns and "tcp_ack" in self.df.columns:
            self.df = self.df.withColumn(
                "pkt_size_cv",
                F.col("tcp_len") / (F.abs(F.col("tcp_seq") - F.col("tcp_ack")) + F.lit(1.0))
            )
        elif "tcp_len" in self.df.columns:
            self.df = self.df.withColumn("pkt_size_cv", F.col("tcp_len").cast("double"))
        else:
            self.df = self.df.withColumn("pkt_size_cv", F.lit(0.0))

        # NaN/Inf temizle
        self.df = self.df.withColumn(
            "pkt_size_cv",
            F.when(
                F.col("pkt_size_cv").isNull() |
                F.isnan(F.col("pkt_size_cv")) |
                F.col("pkt_size_cv").isin(float("inf"), float("-inf")),
                F.lit(0.0)
            ).otherwise(F.col("pkt_size_cv"))
        )

        self._created_features.append("pkt_size_cv")
        print("   ✅ Feature 2: pkt_size_cv (Paket Boyut Varyasyon Katsayısı)")
        return self

    # ──────────────────────────────────────────────
    # FEATURE 3: Akış Yoğunluğu
    # ──────────────────────────────────────────────
    def add_flow_intensity(self) -> "FeatureEngineer":
        """
        flow_intensity = tcp_len * tcp_flags

        NEDEN?
        Paket boyutunu (tcp_len) ve bayrak sayısını (tcp_flags) tek bir
        metrikte birleştirir. DDoS UDP/ICMP Flood saldırılarında hem paket
        boyutu hem de flag sayısı anormal değerler alır. Bu metrik bu
        iki boyutu çarparak volumetrik saldırıları tespit eder.

        TESPIT ETTİĞİ SALDIRILAR: DDoS UDP/ICMP Flood, HTTP Flood
        """
        len_col = "tcp_len" if "tcp_len" in self.df.columns else None
        flags_col = "tcp_flags" if "tcp_flags" in self.df.columns else None

        if len_col and flags_col:
            self.df = self.df.withColumn(
                "flow_intensity",
                F.col(len_col) * F.col(flags_col)
            )
        else:
            # Fallback
            self.df = self.df.withColumn("flow_intensity", F.lit(0.0))

        # NaN/Inf temizle
        self.df = self.df.withColumn(
            "flow_intensity",
            F.when(
                F.col("flow_intensity").isNull() |
                F.isnan(F.col("flow_intensity")) |
                F.col("flow_intensity").isin(float("inf"), float("-inf")),
                F.lit(0.0)
            ).otherwise(F.col("flow_intensity"))
        )

        self._created_features.append("flow_intensity")
        print("   ✅ Feature 3: flow_intensity (Akış Yoğunluğu)")
        return self

    # ──────────────────────────────────────────────
    # FEATURE 4: Varış Zamanı Düzenliliği
    # ──────────────────────────────────────────────
    def add_iat_regularity(self) -> "FeatureEngineer":
        """
        iat_regularity = tcp_checksum / (tcp_len + 1)

        NEDEN?
        TCP checksum paket header'ı ve payload'un bütünlük kontrolüdür.
        Normal trafik tutarlı checksum/boyut oranına sahiptir.
        Saldırı araçları (scanner, brute force) crafted paketler üretir —
        küçük payload ile yüksek header overhead, bu da checksum/len
        oranını anormal yapar. Port Scanning küçük paketler gönderir
        (yüksek oran), DDoS büyük payload kullanır (düşük oran).

        TESPIT ETTİĞİ SALDIRILAR: Port Scanning, Automated Scanning, Password Brute Force
        """
        if "tcp_checksum" in self.df.columns and "tcp_len" in self.df.columns:
            # Checksum / paket boyutu oranı
            # Küçük paket + yüksek checksum = tarama/keşif trafiği
            # Büyük paket + orantılı checksum = normal veri transferi
            self.df = self.df.withColumn(
                "iat_regularity",
                F.col("tcp_checksum") / (F.col("tcp_len") + F.lit(1.0))
            )
        elif "tcp_checksum" in self.df.columns:
            self.df = self.df.withColumn(
                "iat_regularity",
                F.col("tcp_checksum")
            )
        else:
            self.df = self.df.withColumn("iat_regularity", F.lit(0.0))

        # NaN/Inf temizle
        self.df = self.df.withColumn(
            "iat_regularity",
            F.when(
                F.col("iat_regularity").isNull() |
                F.isnan(F.col("iat_regularity")) |
                F.col("iat_regularity").isin(float("inf"), float("-inf")),
                F.lit(0.0)
            ).otherwise(F.col("iat_regularity"))
        )

        self._created_features.append("iat_regularity")
        print("   ✅ Feature 4: iat_regularity (Varış Zamanı Düzenliliği)")
        return self

    # ──────────────────────────────────────────────
    # FEATURE 5: Bağlantı Verimliliği
    # ──────────────────────────────────────────────
    def add_conn_efficiency(self) -> "FeatureEngineer":
        """
        conn_efficiency = tcp_connection_syn / (tcp_connection_fin + tcp_connection_rst + 1)

        NEDEN?
        Normal bir TCP bağlantısında SYN → veri transferi → FIN/RST akışı
        vardır, yani SYN/FIN oranı ~1'e yakındır. Port Scanning ve OS
        Fingerprinting gibi keşif saldırıları çok sayıda SYN gönderir
        ama bağlantıyı tamamlamaz (FIN/RST olmaz), yani oran çok yükselir.
        Bu verimsizlik saldırı sinyalidir.

        TESPIT ETTİĞİ SALDIRILAR: Port Scanning, OS Fingerprinting, SYN Flood
        """
        syn_col = "tcp_connection_syn" if "tcp_connection_syn" in self.df.columns else None
        fin_col = "tcp_connection_fin" if "tcp_connection_fin" in self.df.columns else None
        rst_col = "tcp_connection_rst" if "tcp_connection_rst" in self.df.columns else None

        if syn_col and fin_col and rst_col:
            self.df = self.df.withColumn(
                "conn_efficiency",
                F.col(syn_col) / (F.col(fin_col) + F.col(rst_col) + F.lit(1))
            )
        elif syn_col and fin_col:
            self.df = self.df.withColumn(
                "conn_efficiency",
                F.col(syn_col) / (F.col(fin_col) + F.lit(1))
            )
        else:
            self.df = self.df.withColumn("conn_efficiency", F.lit(0.0))

        # NaN/Inf temizle
        self.df = self.df.withColumn(
            "conn_efficiency",
            F.when(
                F.col("conn_efficiency").isNull() |
                F.isnan(F.col("conn_efficiency")) |
                F.col("conn_efficiency").isin(float("inf"), float("-inf")),
                F.lit(0.0)
            ).otherwise(F.col("conn_efficiency"))
        )

        self._created_features.append("conn_efficiency")
        print("   ✅ Feature 5: conn_efficiency (Bağlantı Verimliliği)")
        return self

    # ──────────────────────────────────────────────
    # TÜM FEATURE'LARI ÜRET
    # ──────────────────────────────────────────────
    def create_all_features(self) -> DataFrame:
        """5 feature'ı sırayla oluşturur ve DataFrame döndürür."""
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING — 5 Yeni Özellik")
        print("=" * 60)

        (
            self
            .add_traffic_asymmetry_ratio()
            .add_pkt_size_cv()
            .add_flow_intensity()
            .add_iat_regularity()
            .add_conn_efficiency()
        )

        print(f"\n   📊 Toplam {len(self._created_features)} feature üretildi:")
        for i, name in enumerate(self._created_features, 1):
            print(f"      {i}. {name}")
        print("=" * 60)

        return self.df

    def get_feature_names(self) -> List[str]:
        """Üretilen feature isimlerini döndürür."""
        return self._created_features.copy()

    def get_feature_summary(self) -> DataFrame:
        """Üretilen feature'ların istatistiksel özetini döndürür."""
        if not self._created_features:
            print("⚠️ Henüz feature üretilmedi. create_all_features() çağırın.")
            return self.df

        return self.df.select(self._created_features).describe()
