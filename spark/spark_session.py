"""
SparkSession Factory
====================
Projenin tüm bileşenleri tarafından kullanılan merkezi SparkSession oluşturucu.
Delta Lake ve Kafka connector JAR'ları ile yapılandırılmış.

Kullanım:
    from spark.spark_session import get_spark
    spark = get_spark()
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "IoT-Intrusion-Detection") -> SparkSession:
    """
    Delta Lake ve Kafka desteği ile yapılandırılmış SparkSession döndürür.

    Args:
        app_name: Spark uygulamasının adı (Spark UI'da görünür)

    Returns:
        SparkSession: Yapılandırılmış Spark oturumu
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        # ── Delta Lake Extensions ──
        # JAR'lar Dockerfile build sırasında /opt/bitnami/spark/jars/ altına kopyalanır.
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        # ── Memory Ayarları ──
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        # ── Delta Lake Performans ──
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .getOrCreate()
    )

    # Log seviyesini azalt (konsol temiz kalsın)
    spark.sparkContext.setLogLevel("WARN")

    return spark
