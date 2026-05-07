# IoT Network Intrusion Detection Pipeline

Bu proje, IoT ve IIoT cihazları üzerinden toplanan Edge-IIoTset siber güvenlik veri setini kullanarak ağ trafiği anomali tespitini amaçlamaktadır. Kafka, Apache Spark, Delta Lake ve MLflow teknolojilerini kullanan uçtan uca bir Büyük Veri (Big Data) ardışık düzeni (pipeline) içerir.

## 🚀 Başlangıç

Sistemi Docker üzerinden tek komutla ayağa kaldırabilirsiniz. Docker ortamı Spark (Master & Worker), JupyterLab, Kafka, Zookeeper ve MLflow'u içerir.

```bash
docker compose up -d
```
> *Servisler başladıktan sonra:*
> - **JupyterLab:** `http://localhost:8888`
> - **Spark UI:** `http://localhost:8080`
> - **MLflow UI:** `http://localhost:5000`

---

## 💾 Veri Setini İndirme (Kaggle)

Projede Kaggle'daki **Edge-IIoTset** veri setini kullanıyoruz. Veri seti oldukça büyük olduğu için depolama alanını doldurmamak adına GitHub'a yüklenmez. Projeyi kendi bilgisayarınızda çalıştırırken veriyi indirmek için aşağıdaki adımları takip etmelisiniz:

### 1. Kaggle API Ayarları
1. [Kaggle.com](https://www.kaggle.com/)'a giriş yapın ve sağ üstteki profilinizden **Settings (Ayarlar)** menüsüne gidin.
2. Sayfayı aşağı kaydırarak **API** başlığı altından `Create New Token` butonuna tıklayın.
3. İndirilen `kaggle.json` dosyasını bilgisayarınızdaki şu konuma yerleştirin:  
   👉 `C:\Users\KULLANICI_ADINIZ\.kaggle\kaggle.json`

### 2. İndirme Komutları
Proje klasöründeyken terminali açıp sırasıyla şu komutları çalıştırın:

```bash
# 1. Kaggle Python aracını bilgisayara kurun
pip install kaggle

# 2. Projenin içine verinin bulunacağı ham veri (raw) klasörünü oluşturun
mkdir -p data\raw

# 3. Veri setini doğrudan data/raw klasörüne indirin ve zip'ten çıkarın
kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -p ./data/raw --unzip
```

*Veri başarıyla indirildiğinde `data/raw/` klasörü içinde analiz edilmeye hazır `.csv` dosyasını göreceksiniz.*

---

## 🔄 Streaming Pipeline Çalıştırma

Tüm Docker servisleri ayakta olduktan sonra (`docker compose up -d`), aşağıdaki adımları sırasıyla uygulayın:

### 1. Kafka Topic Oluşturma

```bash
docker exec -it kafka kafka-topics.sh --create --topic iot-network-traffic --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
```

Topic'in başarıyla oluştuğunu doğrulamak için:

```bash
docker exec -it kafka kafka-topics.sh --list --bootstrap-server localhost:9092
```

### 2. Kafka Producer Başlatma

Producer, CSV dosyasındaki ağ trafiği verilerini Kafka'ya gerçek zamanlı simülasyon olarak gönderir. `--rate` parametresi ile saniyedeki mesaj sayısını ayarlayabilirsiniz:

```bash
docker compose up kafka-producer
```

> **Not:** Producer varsayılan olarak saniyede 50 mesaj gönderir. Hızı değiştirmek isterseniz `docker-compose.yml` içindeki producer servisine `command: python kafka/producer.py --rate 100` gibi bir parametre ekleyebilirsiniz.

### 3. Spark Streaming Pipeline (Bronze → Silver → Gold)

Ayrı bir terminal açarak Spark Structured Streaming pipeline'ını başlatın. Bu komut Bronze, Silver ve Gold katmanlarını sırasıyla başlatır:

```bash
docker exec -it spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.2 /opt/bitnami/spark/spark/run_streaming_pipeline.py
```

> Sadece Bronze katmanını test etmek isterseniz:
> ```bash
> docker exec -it spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.2 /opt/bitnami/spark/spark/run_bronze.py
> ```

Pipeline başarıyla başladığında konsolda şu mesajları görmelisiniz:
```
[1/3] Starting Bronze Layer Ingestion...
[2/3] Starting Silver Layer (Cleaning & Parsing)...
[3/3] Starting Gold Layer (Feature Engineering)...
✅ Tüm streaming katmanları (Bronze -> Silver -> Gold) başlatıldı.
```

Durdurmak için `Ctrl+C` kullanın.

### 4. Pipeline Doğrulama

Pipeline çalıştıktan sonra Delta tablolarını JupyterLab (`http://localhost:8888`) üzerinden sorgulayarak veri akışını doğrulayabilirsiniz:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Verification") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Katman bazında kayıt sayıları (Bronze ≥ Silver ≥ Gold olmalı)
for layer, path in [
    ("Bronze", "/opt/bitnami/spark/delta-storage/bronze/network_traffic"),
    ("Silver", "/opt/bitnami/spark/delta-storage/silver/network_traffic"),
    ("Gold",   "/opt/bitnami/spark/delta-storage/gold/ml_ready")
]:
    df = spark.read.format("delta").load(path)
    print(f"{layer}: {df.count():,} kayıt")
```

---

## 📊 Servis Adresleri

| Servis | Adres |
|---|---|
| JupyterLab | `http://localhost:8888` |
| Spark Master UI | `http://localhost:8080` |
| MLflow UI | `http://localhost:5000` |
| Kafka (dış erişim) | `localhost:29092` |

---

## 🛠️ Sorun Giderme

- **Kafka bağlantı hatası:** Kafka container'ının tamamen başladığından emin olun (`docker logs kafka`). Başlatma 10-15 saniye sürebilir.
- **Spark "package not found" hatası:** İlk çalıştırmada JAR indirmesi zaman alabilir, internet bağlantınızı kontrol edin.
- **Delta tablolar boş:** Producer'ın çalıştığından ve Kafka'ya mesaj gönderdiğinden emin olun (`docker logs kafka-producer`).
- **Memory hataları:** `docker-compose.yml`'daki `SPARK_WORKER_MEMORY` değerini artırın veya bilgisayarınızda Docker'a ayrılan RAM'i yükseltin.
