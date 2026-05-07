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
    
    # Producer'ın payload'a eklediği mandatory field'ları şemaya dahil edelim
    schema.add(StructField("flow_id", StringType(), True))
    schema.add(StructField("source_ip", StringType(), True))
    schema.add(StructField("dest_ip", StringType(), True))
    schema.add(StructField("timestamp", StringType(), True))
    schema.add(StructField("attack_type", StringType(), True))
    
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
    
    # 3. Temizleme İşlemleri (DataCleaner mantığının Streaming versiyonu)
    
    # A. Drop unnecessary columns
    drop_cols = [
        "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4",
        "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused"
    ]
    existing_drop = [c for c in drop_cols if c in parsed_df.columns]
    clean_df = parsed_df.drop(*existing_drop)
    
    # B. Infinity değerleri null yap
    numeric_cols = [col_name for col_name, dtype in clean_df.dtypes if dtype in ("double", "float")]
    for col_name in numeric_cols:
        clean_df = clean_df.withColumn(
            col_name,
            F.when(
                F.col(f"`{col_name}`").isin(float("inf"), float("-inf")),
                None
            ).otherwise(F.col(f"`{col_name}`"))
        )
        
    # C. Eksik Değerleri Doldur
    num_cols = [c for c, d in clean_df.dtypes if d in ("double", "float", "int", "bigint", "long")]
    str_cols = [c for c, d in clean_df.dtypes if d == "string"]
    fill_values = {f"`{c}`": 0.0 for c in num_cols}
    fill_values.update({f"`{c}`": "unknown" for c in str_cols})
    clean_df = clean_df.fillna(fill_values)
    
    # D. Veri tiplerini dönüştür (Attack_label ve portlar)
    if "Attack_label" in clean_df.columns:
        clean_df = clean_df.withColumn("Attack_label", F.col("Attack_label").cast("integer"))
    port_cols = [c for c in clean_df.columns if "port" in c.lower()]
    for c in port_cols:
        clean_df = clean_df.withColumn(c, F.col(f"`{c}`").cast("integer"))
        
    # E. Streaming Drop Duplicates (Watermark gerekli!)
    # Verilerin son 1 dakika içinde geliş zamanlarına göre kopya olup olmadığına bakar
    # 1 dakikadan eski veriler hafızadan (state) silinir.
    clean_df = (
        clean_df
        .withWatermark("ingestion_time", "1 minute")
        .dropDuplicates(["ingestion_time", "flow_id"]) # Aynı saniyede aynı flow_id gelirse kopyadır
    )
    
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
