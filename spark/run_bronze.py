import os
import sys

# Proje dizinini Python path'e ekleyelim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.spark_session import get_spark
from spark.streaming.read_kafka import get_kafka_stream
from spark.streaming.bronze_layer import write_to_bronze

def main():
    print("Starting SparkSession...")
    spark = get_spark("Bronze-Layer-Ingestion")
    
    print("Connecting to Kafka...")
    # Kafka broker adresini docker-compose network içindeki isme göre verdik: kafka:9092
    df_stream = get_kafka_stream(spark, brokers="kafka:9092", topic="iot-network-traffic")
    
    print("Writing stream to Bronze Layer...")
    # Docker ortamı içinden çalıştırıldığını varsaydığımız için path /opt/bitnami/spark/delta-storage
    query = write_to_bronze(df_stream, base_path="/opt/bitnami/spark/delta-storage")
    
    # Process continuously
    query.awaitTermination()

if __name__ == "__main__":
    main()
