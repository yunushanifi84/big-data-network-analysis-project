# 🛡️ IoT Intrusion Detection — Streamlit Dashboard

Big Data dönem projesinin **Adım 7** (Görselleştirme & Dashboard) çıktısı.
Kafka → Spark → Delta Lake → MLflow pipeline'ını ve 5 modelli karşılaştırmayı
tek bir modern arayüzde sunar.

## Sayfalar

| Sayfa | İçerik |
|---|---|
| 📊 Genel Bakış | Pipeline mimari diyagramı, katman durumu, model özetleri, türetilmiş feature kartları |
| 🌊 Veri Akışı | Bronze / Silver / Gold detayları, Sankey akış hacmi, çalıştırma komutları |
| 🔍 EDA | Saldırı tipi dağılımı (bar + pie), zaman serisi, eksik değer, histogramlar, korelasyon |
| ⚙️ Feature Engineering | 5 özellik için formül + iş mantığı + violin dağılımı |
| 🤖 Model Karşılaştırma | Grouped bar (acc/F1/precision/recall), radar, tablo, MLflow run history |
| 🏆 En İyi Model | Confusion matrix, per-class metrik, ROC, feature importance |

## Çalıştırma

### Docker (önerilen)

```bash
docker compose up -d streamlit-dashboard
```

→ http://localhost:8501

### Lokal

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Veri kaynakları

Dashboard şu yolları **canlı** okur (cache TTL: 60–300s):

- `mlruns/mlflow.db` — MLflow SQLite (run, metric, tag, param)
- `delta-storage/{bronze,silver,gold}/...` — Delta Lake katmanları
- `ml/*_feature_importance.csv` — Eğitim sonrası feature importance çıktıları

Yollar `MLFLOW_DB`, `DELTA_ROOT`, `ML_DIR` ortam değişkenleri ile değiştirilebilir.

## Notlar

- **MLflow tag gerekli**: Bir modelin "Model Karşılaştırma" sayfasında otomatik
  görünebilmesi için MLflow run'ında `model_type` tag'inin set edilmiş olması gerekir
  (örn. `"RandomForest"`, `"GradientBoosted"`). Şu an sadece Logistic Regression
  run'larında bu tag mevcut — diğer modelleri yeniden eğitince dashboard otomatik dolar.
- Veriler **örneklenir** (Gold için max 80K satır) — büyük tablolarda da hızlı kalır.
