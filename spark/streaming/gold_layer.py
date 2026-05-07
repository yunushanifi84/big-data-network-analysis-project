import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark.preprocessing.feature_engineering import FeatureEngineer

def write_to_gold(spark: SparkSession, base_path: str = "/opt/bitnami/spark/delta-storage") -> None:
    """
    Silver katmanından temizlenmiş verileri okur, 5 zorunlu feature'ı üretir
    ve ML'e hazır (Gold Layer) olarak Delta formatında yazar.
    """
    
    silver_path = f"{base_path}/silver/network_traffic"
    gold_path = f"{base_path}/gold/ml_ready"
    checkpoint_path = f"{base_path}/checkpoints/gold"
    
    # 1. Silver'dan streaming okuma
    silver_stream = (
        spark.readStream
        .format("delta")
        .load(silver_path)
    )
    
    # 2. Feature Engineering (Adım 5'te yazılan sınıfı aynen kullanıyoruz!)
    # Çünkü yazılan feature formülleri (withColumn) streaming ile %100 uyumlu.
    engineer = FeatureEngineer(spark, silver_stream)
    gold_stream = engineer.create_all_features()
    
    # Not: Vektörleştirme (VectorAssembler) işlemleri Spark Structured Streaming'de 
    # bazen ML nesneleri kullanıldığında zorlayıcı olabilir. 
    # Ancak feature hesaplamaları doğrudan withColumn olduğu için mükemmel çalışır.
    
    # 3. Gold Katmanına Yazma
    query = (
        gold_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="30 seconds")
        .start(gold_path)
    )
    
    return query
