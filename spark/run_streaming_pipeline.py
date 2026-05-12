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


def wait_for_delta_table(path: str, timeout: int = 120, poll_interval: int = 5) -> bool:
    """
    Delta tablosunun oluşmasını bekler.
    Önceki katmanın en az bir batch yazması gerekiyor, yoksa
    sonraki katman 'Table schema is not set' hatası alır.
    
    Args:
        path: Delta tablosunun yolu
        timeout: Maksimum bekleme süresi (saniye)
        poll_interval: Kontrol aralığı (saniye)
    
    Returns:
        True: Tablo bulundu, False: Timeout
    """
    delta_log = os.path.join(path, "_delta_log")
    elapsed = 0
    while elapsed < timeout:
        if os.path.exists(delta_log) and any(
            f.endswith(".json") for f in os.listdir(delta_log)
        ):
            print(f"   ✅ Delta tablosu hazır: {path}")
            return True
        print(f"   ⏳ Delta tablosu bekleniyor ({elapsed}s / {timeout}s): {path}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    print(f"   ❌ TIMEOUT! Delta tablosu oluşmadı: {path}")
    return False


def main():
    print("Starting SparkSession for End-to-End Streaming...")
    spark = get_spark("IoT-Streaming-Pipeline")
    
    base_path = "/opt/bitnami/spark/delta-storage"
    
    print("\n[1/3] Starting Bronze Layer Ingestion...")
    # 1. Kafka -> Bronze
    df_stream = get_kafka_stream(spark, brokers="kafka:9092", topic="iot-network-traffic")
    bronze_query = write_to_bronze(df_stream, base_path=base_path)
    
    # Bronze'un Delta tablosunu oluşturmasını bekle
    bronze_path = f"{base_path}/bronze/network_traffic"
    print("\n   Bronze Delta tablosunun oluşmasını bekliyoruz...")
    if not wait_for_delta_table(bronze_path, timeout=120):
        print("   ❌ Bronze tablosu oluşturulamadı! Kafka producer çalışıyor mu kontrol edin.")
        spark.stop()
        sys.exit(1)
    
    print("\n[2/3] Starting Silver Layer (Cleaning & Parsing)...")
    # 2. Bronze -> Silver
    silver_query = write_to_silver(spark, base_path=base_path)
    
    # Silver'ın Delta tablosunu oluşturmasını bekle
    silver_path = f"{base_path}/silver/network_traffic"
    print("\n   Silver Delta tablosunun oluşmasını bekliyoruz...")
    if not wait_for_delta_table(silver_path, timeout=300):
        print("   ❌ Silver tablosu oluşturulamadı! Silver layer loglarını kontrol edin.")
        spark.stop()
        sys.exit(1)
    
    print("\n[3/3] Starting Gold Layer (Feature Engineering)...")
    # 3. Silver -> Gold
    gold_query = write_to_gold(spark, base_path=base_path)
    
    print("\n✅ Tüm streaming katmanları (Bronze -> Silver -> Gold) başlatıldı.")
    print("Pipeline arka planda çalışmaya devam ediyor. Çıkmak için Ctrl+C kullanın.")
    
    # Herhangi birinin çökmesini bekler
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
