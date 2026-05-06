from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import from_json, col

def get_kafka_stream(spark: SparkSession, 
                     brokers: str = "kafka:9092", 
                     topic: str = "iot-network-traffic") -> DataFrame:
    """
    Kafka'dan streaming verisini okur, binary 'value' kolonunu JSON'a parse eder 
    ve anlamlı bir yapıya (StructType) dönüştürerek DataFrame döndürür.
    """
    
    # 1. Kafka'dan okuma
    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", brokers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    
    # 2. Binary olan value kolonunu String'e dönüştürme ve partition için zamanı tutma
    # Kafka'nın kendi timestamp kolonunu (mesaj geliş zamanı) 'ingestion_time' olarak adlandıralım
    # Gelen json_payload ham olarak (Bronze prensibine uygun) kalır.
    parsed_df = raw_df.selectExpr(
        "CAST(value AS STRING) as json_payload",
        "timestamp as ingestion_time"
    )
    
    return parsed_df

