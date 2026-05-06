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
