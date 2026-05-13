# IoT Network Intrusion Detection Pipeline

Bu proje, IoT ve IIoT cihazları üzerinden toplanan **Edge-IIoTset** siber güvenlik veri setini kullanarak ağ trafiği üzerinde **saldırı tipi sınıflandırması** yapmayı amaçlamaktadır. Kafka ile gerçek zamanlı veri akışı, Apache Spark ile dağıtık işleme, Delta Lake ile katmanlı depolama (Medallion Architecture) ve MLflow ile deney takibi yapan uçtan uca bir **Büyük Veri (Big Data)** pipeline'ı içerir.

## � Proje Ekibi

| Öğrenci No | Ad Soyad |
|---|---|
| 220201018 | Abdulrahman Dülek |
| 220201061 | Eyüp Ensar Kara |
| 220201083 | Yunus Hanifi Öztürk |

## �📋 Proje Özeti

| Bileşen | Teknoloji |
|---|---|
| Veri Akışı | Kafka + Zookeeper |
| Dağıtık İşleme | Apache Spark (Structured Streaming + MLlib) |
| Depolama | Delta Lake (Bronze → Silver → Gold) |
| ML Deney Takibi | MLflow |
| Dashboard | Streamlit (çok sayfalı) |
| Notebook Analizi | JupyterLab |
| Orkestrasyon | Docker Compose |

### Sınıflandırma Yaklaşımı

Tüm modeller **multi-class** sınıflandırma yapar — `Attack_type` kolonu hedef değişkendir. Modeller ağ trafiğinin **hangi saldırı türüne** ait olduğunu (Normal, DDoS_HTTP, Port_Scanning, MITM, Ransomware, vb.) tahmin eder.

| # | Model | Açıklama |
|---|---|---|
| 1 | **Logistic Regression** | Multinomial LR (`family="multinomial"`) |
| 2 | **Decision Tree** | Yorumlanabilir karar ağacı |
| 3 | **Random Forest** | Bagging ensemble (100+ ağaç) |
| 4 | **Gradient Boosted Trees** | `OneVsRest` wrapper ile multi-class (her sınıf için ayrı binary GBT) |
| 5 | **Naive Bayes** | MinMaxScaler + multinomial NB (baseline) |

---

## 🚀 Başlangıç

Sistemi Docker üzerinden tek komutla ayağa kaldırabilirsiniz:

```bash
docker compose up -d
```

> *Servisler başladıktan sonra:*
> - **Streamlit Dashboard:** `http://localhost:8501`
> - **JupyterLab:** `http://localhost:8888`
> - **Spark UI:** `http://localhost:8080`
> - **MLflow UI:** `http://localhost:5000`

---

## 💾 Veri Setini İndirme (Kaggle)

Projede Kaggle'daki **Edge-IIoTset** veri setini kullanıyoruz. Veri seti oldukça büyük olduğu için GitHub'a yüklenmez.

### 1. Kaggle API Ayarları
1. [Kaggle.com](https://www.kaggle.com/)'a giriş yapın ve sağ üstteki profilinizden **Settings (Ayarlar)** menüsüne gidin.
2. Sayfayı aşağı kaydırarak **API** başlığı altından `Create New Token` butonuna tıklayın.
3. İndirilen `kaggle.json` dosyasını bilgisayarınızdaki şu konuma yerleştirin:  
   👉 `C:\Users\KULLANICI_ADINIZ\.kaggle\kaggle.json`

### 2. İndirme Komutları

```bash
pip install kaggle
mkdir -p data\raw
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

Producer, CSV dosyasındaki ağ trafiği verilerini Kafka'ya gerçek zamanlı simülasyon olarak gönderir:

```bash
docker compose up --build kafka-producer
```

> **Not:** Producer parametreleri `docker-compose.yml` içindeki `kafka-producer.command` alanından yönetilir.
> - Gecikmesiz tam hız: `--no-delay`
> - Mesaj limiti: `--max-messages 5000` (0 = sınırsız)
> - Hız limiti: `--rate 100` (no-delay kapalıysa geçerli)

### 3. Spark Streaming Pipeline (Bronze → Silver → Gold)

Ayrı bir terminal açarak Spark Structured Streaming pipeline'ını başlatın:

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

---

## 📊 Servis Adresleri

| Servis | Adres |
|---|---|
| Streamlit Dashboard | `http://localhost:8501` |
| JupyterLab | `http://localhost:8888` |
| Spark Master UI | `http://localhost:8080` |
| MLflow UI | `http://localhost:5000` |
| Kafka (dış erişim) | `localhost:29092` |

---

## 🤖 Model Eğitimleri (Multi-class)

Tüm modeller Gold katmanındaki temiz ve zenginleştirilmiş veriyi kullanarak `Attack_type` sınıflandırması yapar. Her model CrossValidator ile hiperparametre optimizasyonu uygular ve sonuçları MLflow'a loglar.

Model eğitimi için **iki yöntem** sunulmaktadır:

| Yöntem | Açıklama | Ne Zaman Kullanmalı? |
|---|---|---|
| **A) CLI (spark-submit)** | Terminal üzerinden tek komutla eğitim | Hızlı, otomatize edilebilir, production odaklı |
| **B) Jupyter Notebook** | Adım adım interaktif eğitim | Keşifsel analiz, görselleştirme, eğitim sürecini anlama |

> 💡 Her iki yöntem de aynı model mantığını kullanır ve sonuçları MLflow'a loglar.

### 0. Kurulum Doğrulama

```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/validate_setup.py
```

### 1. Logistic Regression (Multinomial)

**A) CLI:**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/01_logistic_regression.py
```

**B) Notebook:** [`notebooks/03_logistic_regression.ipynb`](notebooks/03_logistic_regression.ipynb) → JupyterLab: `http://localhost:8888`

### 2. Decision Tree

**A) CLI:**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/02_decision_tree.py
```

**B) Notebook:** [`notebooks/04_decision_tree.ipynb`](notebooks/04_decision_tree.ipynb) → JupyterLab: `http://localhost:8888`

### 3. Random Forest

**A) CLI:**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/03_random_forest.py
```

**B) Notebook:** [`notebooks/05_random_forest.ipynb`](notebooks/05_random_forest.ipynb) → JupyterLab: `http://localhost:8888`

### 4. GBT (OneVsRest)

> ⚠️ GBT doğrudan multi-class desteklemez; `OneVsRest` wrapper ile her sınıf için ayrı binary GBT eğitilir. Diğer modellere göre ~N kat daha yavaştır.

**A) CLI:**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/04_gbt.py
```

**B) Notebook:** [`notebooks/06_gbt.ipynb`](notebooks/06_gbt.ipynb) → JupyterLab: `http://localhost:8888`

### 5. Naive Bayes

**A) CLI:**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/05_naive_bayes.py
```

**B) Notebook:** [`notebooks/07_naive_bayes.ipynb`](notebooks/07_naive_bayes.ipynb) → JupyterLab: `http://localhost:8888`

### Ortak CLI Parametreleri

Tüm model script'leri aşağıdaki argümanları kabul eder:

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| `--fast` | Örneklem ile hızlı eğitim (test amaçlı) | Kapalı |
| `--sample-size N` | Hızlı modda kullanılacak satır sayısı | 30000 |
| `--cv-mode quick\|full` | CV grid kapsamı | `quick` |
| `--cv-parallelism N` | CrossValidator parallelism | 2 |

**Örnek (hızlı test):**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/03_random_forest.py --fast --sample-size 50000
```

**Örnek (kapsamlı eğitim):**
```bash
docker exec spark-master spark-submit --packages io.delta:delta-core_2.12:2.4.0 /opt/bitnami/spark/ml/03_random_forest.py --cv-mode full
```

### MLflow Çıktıları

Her eğitim (CLI veya Notebook) şu artifact'ları MLflow'a loglar:
- **Metrikler:** accuracy, f1_score, precision, recall, log_loss (weighted, multi-class)
- **Parametreler:** tüm hiperparametreler + CV konfigürasyonu
- **Artifact'lar:** `confusion_matrix.csv` (NxN), `label_index_mapping.csv`, `cv_results.csv`, feature importance chart (PNG + CSV)
- **Tag'ler:** `classification_type=multiclass`, `label_column=Attack_type`, `num_classes=N`

---

## 📈 Streamlit Dashboard

Çok sayfalı interaktif dashboard:

```
http://localhost:8501
```

| Sayfa | İçerik |
|---|---|
| Genel Bakış | Sistem durumu, servis sağlığı |
| Veri Akışı | Kafka → Delta Lake pipeline izleme |
| EDA | Keşifsel veri analizi ve görselleştirmeler |
| Feature Engineering | Gold katmanı özellik mühendisliği detayları |
| Model Karşılaştırma | 5 modelin yan yana multi-class karşılaştırması (F1 sıralı) |
| En İyi Model | Şampiyon model seçimi ve production tavsiyesi |

---

## 📂 Proje Yapısı

```
big-data-project/
├── kafka/                      # Kafka producer & consumer
│   ├── producer.py
│   └── consumer_test.py
├── spark/
│   ├── preprocessing/          # Veri temizleme, kalite kontrolü
│   │   ├── data_cleaner.py
│   │   ├── data_quality.py
│   │   ├── feature_engineering.py
│   │   └── run_pipeline.py
│   ├── streaming/              # Bronze / Silver / Gold katmanları
│   │   ├── bronze_layer.py
│   │   ├── silver_layer.py
│   │   ├── gold_layer.py
│   │   └── read_kafka.py
│   ├── spark_session.py        # Paylaşılan Spark oturum yönetimi
│   ├── run_bronze.py           # Sadece Bronze test script'i
│   └── run_streaming_pipeline.py  # Uçtan uca streaming
├── ml/
│   ├── utils.py                # Ortak ML utility'leri (pipeline, eval, mlflow)
│   ├── 01_logistic_regression.py
│   ├── 02_decision_tree.py
│   ├── 03_random_forest.py
│   ├── 04_gbt.py               # OneVsRest wrapper
│   ├── 05_naive_bayes.py
│   ├── validate_setup.py
│   ├── check_runs.py           # MLflow run kontrol
│   └── sanity_check.py         # Veri doğrulama
├── streamlit_app/
│   ├── app.py                  # Ana sayfa
│   ├── data_loader.py          # MLflow DB & Delta okuyucu
│   ├── theme.py                # Koyu tema
│   └── pages/                  # Streamlit çok sayfalı yapı
│       ├── 1_Genel_Bakış.py
│       ├── 2_Veri_Akışı.py
│       ├── 3_EDA.py
│       ├── 4_Feature_Engineering.py
│       ├── 5_Model_Karşılaştırma.py
│       └── 6_En_İyi_Model.py
├── notebooks/                  # Jupyter analiz notebook'ları
│   ├── 01_eda.ipynb
│   ├── 02_data_presentation.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_gbt.ipynb
│   └── 07_naive_bayes.ipynb
├── docs/                       # Proje dokümanları
│   ├── rapor.tex
│   └── data_quality_report.md
├── docker/                     # Dockerfile'lar
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Sorun Giderme

- **Kafka bağlantı hatası:** Kafka container'ının tamamen başladığından emin olun (`docker logs kafka`). Başlatma 10-15 saniye sürebilir.
- **Spark "package not found" hatası:** İlk çalıştırmada JAR indirmesi zaman alabilir, internet bağlantınızı kontrol edin.
- **Delta tablolar boş:** Producer'ın çalıştığından ve Kafka'ya mesaj gönderdiğinden emin olun (`docker logs kafka-producer`).
- **Memory hataları:** `docker-compose.yml`'daki `SPARK_WORKER_MEMORY` değerini artırın veya bilgisayarınızda Docker'a ayrılan RAM'i yükseltin.
- **GBT çok uzun sürüyor:** OneVsRest N kat yavaştır. `--fast --sample-size 100000` ile orta yol deneyin veya `--cv-mode quick` kullanın.
- **Model "MLflow ❌" gösteriyor:** İlgili model henüz eğitilmemiş. Yukarıdaki eğitim komutlarını çalıştırın.
