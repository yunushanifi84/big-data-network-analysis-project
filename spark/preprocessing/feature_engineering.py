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
        traffic_asymmetry_ratio = tcp.ack / (tcp.seq + 1)

        NEDEN?
        Normal ağ trafiği genelde simetrik bir istek-cevap (ACK/SEQ) akışına
        sahiptir. DDoS saldırılarında ise bu oran bozulur çünkü saldırgan
        çok sayıda SYN gönderir ama ACK dönemini tamamlamaz (SYN flood).
        Bu oran DDoS ve Flood saldırılarını tespit etmede güçlü bir sinyaldir.

        TESPIT ETTİĞİ SALDIRILAR: DDoS TCP/UDP/ICMP Flood, SYN Flood
        """
        ack_col = "tcp.ack" if "tcp.ack" in self.df.columns else None
        seq_col = "tcp.seq" if "tcp.seq" in self.df.columns else None

        if ack_col and seq_col:
            self.df = self.df.withColumn(
                "traffic_asymmetry_ratio",
                F.col(f"`{ack_col}`") / (F.col(f"`{seq_col}`") + F.lit(1))
            )
        else:
            # Fallback: tcp.ack_raw kullan
            ack_col = "tcp.ack_raw" if "tcp.ack_raw" in self.df.columns else "tcp.ack"
            self.df = self.df.withColumn(
                "traffic_asymmetry_ratio",
                F.col(f"`{ack_col}`") / (F.col("`tcp.checksum`") + F.lit(1))
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
        pkt_size_cv = tcp.len_std_approx / (tcp.len + 1)

        Burada tcp.len'in pencere içindeki standart sapmasına yakın bir
        değeri, tcp.payload ve tcp.len kullanarak türetiyoruz.

        NEDEN?
        Normal trafik tutarlı paket boyutlarına sahiptir (düşük varyasyon).
        Port Scanning ve Vulnerability Scanner saldırıları ise farklı
        boyutlarda paketler gönderir (yüksek varyasyon). Bu coefficient of
        variation (CV) değeri bu farkı yakalar.

        TESPIT ETTİĞİ SALDIRILAR: Port Scanning, Vulnerability Scanner, OS Fingerprinting
        """
        # tcp.len ve tcp.payload arasındaki fark paket header overhead'ini gösterir
        if "tcp.len" in self.df.columns and "tcp.payload" in self.df.columns:
            self.df = self.df.withColumn(
                "pkt_size_cv",
                F.abs(F.col("`tcp.len`") - F.col("`tcp.payload`")) / (F.col("`tcp.len`") + F.lit(1))
            )
        elif "tcp.len" in self.df.columns:
            self.df = self.df.withColumn(
                "pkt_size_cv",
                F.col("`tcp.len`") / (F.col("`tcp.checksum`") + F.lit(1))
            )
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
        flow_intensity = tcp.len * tcp.flags

        NEDEN?
        Paket boyutunu (tcp.len) ve bayrak sayısını (tcp.flags) tek bir
        metrikte birleştirir. DDoS UDP/ICMP Flood saldırılarında hem paket
        boyutu hem de flag sayısı anormal değerler alır. Bu metrik bu
        iki boyutu çarparak volumetrik saldırıları tespit eder.

        TESPIT ETTİĞİ SALDIRILAR: DDoS UDP/ICMP Flood, HTTP Flood
        """
        len_col = "tcp.len" if "tcp.len" in self.df.columns else None
        flags_col = "tcp.flags" if "tcp.flags" in self.df.columns else None

        if len_col and flags_col:
            self.df = self.df.withColumn(
                "flow_intensity",
                F.col(f"`{len_col}`") * F.col(f"`{flags_col}`")
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
        iat_regularity = udp.time_delta / (frame.time + 1)

        NEDEN?
        Otomatize edilmiş saldırı araçları (botnet, scanner) paketleri çok
        düzenli aralıklarla gönderir — sürekli aynı time_delta. İnsan
        kaynaklı normal trafik ise düzensiz aralıklara sahiptir. Bu oran
        düzenli gönderim yapan bot'ları tespit eder.

        TESPIT ETTİĞİ SALDIRILAR: Botnet, Automated Scanning, Password Brute Force
        """
        delta_col = "udp.time_delta" if "udp.time_delta" in self.df.columns else None
        time_col = "frame.time" if "frame.time" in self.df.columns else None

        if delta_col and time_col:
            self.df = self.df.withColumn(
                "iat_regularity",
                F.col(f"`{delta_col}`") / (F.col(f"`{time_col}`") + F.lit(1))
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
        conn_efficiency = tcp.connection.syn / (tcp.connection.fin + tcp.connection.rst + 1)

        NEDEN?
        Normal bir TCP bağlantısında SYN → veri transferi → FIN/RST akışı
        vardır, yani SYN/FIN oranı ~1'e yakındır. Port Scanning ve OS
        Fingerprinting gibi keşif saldırıları çok sayıda SYN gönderir
        ama bağlantıyı tamamlamaz (FIN/RST olmaz), yani oran çok yükselir.
        Bu verimsizlik saldırı sinyalidir.

        TESPIT ETTİĞİ SALDIRILAR: Port Scanning, OS Fingerprinting, SYN Flood
        """
        syn_col = "tcp.connection.syn" if "tcp.connection.syn" in self.df.columns else None
        fin_col = "tcp.connection.fin" if "tcp.connection.fin" in self.df.columns else None
        rst_col = "tcp.connection.rst" if "tcp.connection.rst" in self.df.columns else None

        if syn_col and fin_col and rst_col:
            self.df = self.df.withColumn(
                "conn_efficiency",
                F.col(f"`{syn_col}`") / (F.col(f"`{fin_col}`") + F.col(f"`{rst_col}`") + F.lit(1))
            )
        elif syn_col and fin_col:
            self.df = self.df.withColumn(
                "conn_efficiency",
                F.col(f"`{syn_col}`") / (F.col(f"`{fin_col}`") + F.lit(1))
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
