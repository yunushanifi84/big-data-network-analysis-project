from pyspark.sql import DataFrame
from pyspark.sql.functions import year, month, dayofmonth

def write_to_bronze(df: DataFrame, base_path: str = "/app/delta-storage") -> None:
    """
    Kafka'dan gelen ham veriyi (Bronze Layer) Delta Lake'e yazar.
    Hiçbir yapısal veya mantıksal değişiklik yapılmadan sadece ham veriler tutulur.
    Partitioning: ingestion_year / ingestion_month / ingestion_day
    """
    
    # Partitioning için zaman kolonlarını üretelim
    df_with_partitions = (
        df.withColumn("ingestion_year", year("ingestion_time"))
          .withColumn("ingestion_month", month("ingestion_time"))
          .withColumn("ingestion_day", dayofmonth("ingestion_time"))
    )
    
    # Delta formatında yazma işlemi
    bronze_path = f"{base_path}/bronze/network_traffic"
    checkpoint_path = f"{base_path}/checkpoints/bronze"
    
    query = (
        df_with_partitions.writeStream
        .format("delta")
        .outputMode("append")
        .partitionBy("ingestion_year", "ingestion_month", "ingestion_day")
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="30 seconds")
        .start(bronze_path)
    )
    
    return query
