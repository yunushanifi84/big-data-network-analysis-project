import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

def get_csv_schema(spark: SparkSession) -> StructType:
    """Producer'ın gönderdiği JSON loglarının şemasını CSV'den bir defaya mahsus çıkarır."""
    from pyspark.sql.types import StructField, StringType
    csv_path = "/opt/bitnami/spark/data/raw/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
    # Sadece ilk 10 satırı okuyup şemayı infer etmek hızlıdır
    temp_df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
    schema = temp_df.schema
    
    # Producer'ın payload'a eklediği mandatory field'ları şemaya dahil edelim.
    # NOT: CSV'de zaten bulunan kolon isimlerini eklememek gerekir,
    # çünkü Spark case-insensitive çalışır ve çakışma (AMBIGUOUS_REFERENCE) yaratır.
    existing_col_names = {f.name.lower() for f in schema.fields}
    extra_fields = [
        StructField("flow_id", StringType(), True),
        StructField("source_ip", StringType(), True),
        StructField("dest_ip", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("attack_type", StringType(), True),
    ]
    for field in extra_fields:
        if field.name.lower() not in existing_col_names:
            schema.add(field)
    
    return schema

def write_to_silver(spark: SparkSession, base_path: str = "/opt/bitnami/spark/delta-storage") -> None:
    """
    Bronze katmanından ham JSON loglarını okur, parse eder,
    temizler (Silver Layer) ve Delta formatında yazar.
    """
    
    bronze_path = f"{base_path}/bronze/network_traffic"
    silver_path = f"{base_path}/silver/network_traffic"
    checkpoint_path = f"{base_path}/checkpoints/silver"
    
    # 1. Bronze'dan streaming okuma
    bronze_stream = (
        spark.readStream
        .format("delta")
        .load(bronze_path)
    )
    
    # 2. JSON'u Parse etme
    json_schema = get_csv_schema(spark)
    
    parsed_df = bronze_stream.withColumn(
        "data", F.from_json(F.col("json_payload"), json_schema)
    ).select("data.*", "ingestion_time")
    
    # --- Noktalı kolon isimlerini düzelt ---
    # CSV'deki kolon isimleri nokta içeriyor (dns.qry.type, tcp.connection.fin vb.).
    # Spark noktaları nested struct erişimi olarak yorumladığından
    # withColumn, fillna gibi işlemler çöküyor.
    # Çözüm: tüm kolon isimlerindeki noktaları alt çizgiyle (_) değiştir.
    for col_name in parsed_df.columns:
        if "." in col_name:
            new_name = col_name.replace(".", "_")
            parsed_df = parsed_df.withColumnRenamed(col_name, new_name)
    
    # 3. Temizleme İşlemleri (DataCleaner mantığının Streaming versiyonu)
    
    # A. Drop unnecessary columns (isimler artık _ ile)
    drop_cols = [
        "arp_dst_proto_ipv4", "arp_opcode", "arp_hw_size", "arp_src_proto_ipv4",
        "icmp_checksum", "icmp_seq_le", "icmp_transmit_timestamp", "icmp_unused"
    ]
    existing_drop = [c for c in drop_cols if c in parsed_df.columns]
    clean_df = parsed_df.drop(*existing_drop)
    
    # B. Infinity değerleri null yap
    numeric_cols = [col_name for col_name, dtype in clean_df.dtypes if dtype in ("double", "float")]
    for col_name in numeric_cols:
        clean_df = clean_df.withColumn(
            col_name,
            F.when(
                F.col(col_name).isin(float("inf"), float("-inf")),
                F.lit(None)
            ).otherwise(F.col(col_name))
        )
        
    # C. Eksik Değerleri Doldur
    clean_df = clean_df.fillna(0.0).fillna("unknown")
    
    # D. Veri tiplerini dönüştür (Attack_label ve portlar)
    if "Attack_label" in clean_df.columns:
        clean_df = clean_df.withColumn("Attack_label", F.col("Attack_label").cast("integer"))
    port_cols = [c for c in clean_df.columns if "port" in c.lower()]
    for c in port_cols:
        clean_df = clean_df.withColumn(c, F.col(c).cast("integer"))
        
    # E. Streaming Drop Duplicates
    # Aynı flow_id tekrar gelirse kopya kabul edilir.
    # Not: Watermark olmadan state süresiz tutulur ancak bu veri seti için (≤200K satır)
    # bellek açısından güvenlidir.
    clean_df = clean_df.dropDuplicates(["flow_id"])
    
    # 4. Silver Katmanına Yazma
    query = (
        clean_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="30 seconds")
        .start(silver_path)
    )
    
    return query
