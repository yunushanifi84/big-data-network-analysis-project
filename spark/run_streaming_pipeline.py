import os
import sys
import time

# Proje dizinini Python path'e ekleyelim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.spark_session import get_spark
from spark.streaming.read_kafka import get_kafka_stream
from spark.streaming.bronze_layer import write_to_bronze
from spark.streaming.silver_layer import write_to_silver
from spark.streaming.gold_layer import write_to_gold

def main():
    print("Starting SparkSession for End-to-End Streaming...")
    spark = get_spark("IoT-Streaming-Pipeline")
    
    base_path = "/opt/bitnami/spark/delta-storage"
    
    print("\n[1/3] Starting Bronze Layer Ingestion...")
    # 1. Kafka -> Bronze
    df_stream = get_kafka_stream(spark, brokers="kafka:9092", topic="iot-network-traffic")
    bronze_query = write_to_bronze(df_stream, base_path=base_path)
    
    # Silver'ın başlaması için Bronze'da biraz veri birikmesi gerekebilir, ama streaming arka planda yürür.
    time.sleep(5)
    
    print("\n[2/3] Starting Silver Layer (Cleaning & Parsing)...")
    # 2. Bronze -> Silver
    silver_query = write_to_silver(spark, base_path=base_path)
    
    time.sleep(5)
    
    print("\n[3/3] Starting Gold Layer (Feature Engineering)...")
    # 3. Silver -> Gold
    gold_query = write_to_gold(spark, base_path=base_path)
    
    print("\n✅ Tüm streaming katmanları (Bronze -> Silver -> Gold) başlatıldı.")
    print("Pipeline arka planda çalışmaya devam ediyor. Çıkmak için Ctrl+C kullanın.")
    
    # Herhangi birinin çökmesini bekler
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
