"""
ML Utility Fonksiyonları
========================
Tüm ML model script'leri tarafından ortak kullanılan yardımcı fonksiyonlar.
- Veri yükleme (Gold Delta Lake)
- Train/Test split (stratified, %80/%20)
- Feature vektörleme (VectorAssembler + StringIndexer)
- Metrik hesaplama (AUC, Accuracy, F1, Precision, Recall)
- MLflow logging wrapper
- Confusion matrix hesaplama

Kullanım:
    from ml.utils import load_gold_data, prepare_features, evaluate_model, log_to_mlflow
"""

import os
import sys
import time
import mlflow
import mlflow.spark
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StringIndexerModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


# ─────────────────────────────────────────────
#  1. MLflow Bağlantısı
# ─────────────────────────────────────────────

def init_mlflow(
    tracking_uri: str = "http://mlflow-server:5000",
    experiment_name: str = "iot_intrusion_detection",
) -> None:
    """
    MLflow tracking URI ve experiment'ı ayarlar.
    Docker container içinde veya dışında çalışabilir.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"✅ MLflow bağlantısı kuruldu: {tracking_uri}")
    print(f"   Experiment: {experiment_name}")


# ─────────────────────────────────────────────
#  2. Veri Yükleme
# ─────────────────────────────────────────────

def load_gold_data(
    spark: SparkSession,
    gold_path: str = "/opt/bitnami/spark/delta-storage/gold/ml_ready",
    log_stats: bool = False,
) -> DataFrame:
    """
    Gold katmanından ML'e hazır Delta verisini batch olarak okur.
    Streaming değil — ML eğitimi için statik DataFrame döner.

    Args:
        spark: SparkSession
        gold_path: Gold Delta tablosunun yolu

    Returns:
        DataFrame: Gold katmanından okunan veri
    """
    print(f"📥 Gold katmanından veri okunuyor: {gold_path}")
    df = spark.read.format("delta").load(gold_path)

    col_count = len(df.columns)
    print(f"   ✅ Okuma tamamlandı. Kolon sayısı: {col_count}")
    if log_stats:
        row_count = df.count()
        print(f"   📊 Satır sayısı: {row_count:,}")

    return df


# ─────────────────────────────────────────────
#  3. Feature Hazırlığı
# ─────────────────────────────────────────────

def get_feature_columns(df: DataFrame) -> list:
    """
    ML için kullanılacak numerik feature kolon isimlerini döndürür.
    Label, string ve metadata kolonlarını hariç tutar.

    Args:
        df: Gold DataFrame

    Returns:
        list: Feature kolon isimleri
    """
    # Hariç tutulacak kolonlar (label, metadata, string tipler)
    exclude_cols = {
        # Label kolonları
        "Attack_label", "Attack_type", "attack_type",
        # Metadata / non-feature kolonları
        "ingestion_time", "timestamp", "flow_id",
        "source_ip", "dest_ip",
        # String tipi olan diğer kolonlar
    }

    # Sadece numerik tipleri al
    numeric_types = {"double", "float", "int", "integer", "long", "short", "bigint"}
    feature_cols = [
        col_name
        for col_name, dtype in df.dtypes
        if dtype in numeric_types and col_name not in exclude_cols
    ]

    print(f"   📊 {len(feature_cols)} numerik feature kolon seçildi.")
    return feature_cols


def prepare_features(
    df: DataFrame,
    label_col: str = "Attack_label",
    feature_cols: list = None,
    log_stats: bool = False,
) -> DataFrame:
    """
    VectorAssembler ile feature vektörü, StringIndexer ile label encoding yapar.

    Binary sınıflandırma için:
    - Attack_label: 0 = Normal, 1 = Attack

    Args:
        df: Gold DataFrame
        label_col: Hedef değişken kolon adı
        feature_cols: Kullanılacak feature kolonları (None ise otomatik seçilir)

    Returns:
        DataFrame: 'features' ve 'label' kolonları eklenmiş DataFrame
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    print(f"🔧 Feature vektörü oluşturuluyor ({len(feature_cols)} feature)...")

    select_exprs = [
        F.coalesce(
            F.when(F.isnan(F.col(c).cast("double")), F.lit(0.0))
             .otherwise(F.col(c).cast("double")),
            F.lit(0.0),
        ).alias(c)
        for c in feature_cols
    ]
    select_exprs.append(F.col(label_col).cast("double").alias("label"))
    df = df.select(*select_exprs)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    assembled_df = assembler.transform(df).select("features", "label")

    assembled_df = assembled_df.filter(F.col("label").isNotNull())

    print("   ✅ Feature hazırlığı tamamlandı (tek select + assembler).")
    if log_stats:
        final_count = assembled_df.count()
        print(f"   📊 Feature sonrası satır sayısı: {final_count:,}")

    return assembled_df


# ─────────────────────────────────────────────
#  4. Train / Test Split
# ─────────────────────────────────────────────

def split_data(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
    log_stats: bool = True,
    stratified: bool = True,
) -> tuple:
    """
    Train/test bölme.
    Varsayılan olarak label bazlı stratified split uygular.

    Args:
        df: Feature ve label kolonları içeren DataFrame
        train_ratio: Eğitim oranı (varsayılan: 0.8)
        seed: Rastgele tohum (tekrarlanabilirlik için)

    Returns:
        (train_df, test_df): Eğitim ve test DataFrame'leri
    """
    if stratified:
        df_with_id = df.withColumn("_row_id", F.monotonically_increasing_id())
        fractions = {
            row["label"]: train_ratio
            for row in df_with_id.select("label").distinct().collect()
        }
        train_df = df_with_id.sampleBy("label", fractions=fractions, seed=seed)
        test_df = df_with_id.join(
            train_df.select("_row_id"), on="_row_id", how="left_anti"
        )
        train_df = train_df.drop("_row_id")
        test_df = test_df.drop("_row_id")
    else:
        test_ratio = 1.0 - train_ratio
        train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)

    if log_stats:
        train_count = train_df.count()
        test_count = test_df.count()
        total = train_count + test_count

        split_mode = "stratified" if stratified else "random"
        print(f"📊 Train/Test split yapıldı (mode={split_mode}, seed={seed}):")
        print(f"   Train: {train_count:,} ({train_count/total*100:.1f}%)")
        print(f"   Test:  {test_count:,} ({test_count/total*100:.1f}%)")

        # Sınıf dağılımını göster
        print("\n   Sınıf dağılımı:")
        for row in df.groupBy("label").count().orderBy("label").collect():
            label_val = row["label"]
            label_name = "Normal" if label_val == 0.0 else "Attack"
            print(f"   {label_name} (label={int(label_val)}): {row['count']:,}")

    return train_df, test_df


def compute_class_weights(df: DataFrame, label_col: str = "label") -> dict:
    """
    Sınıf dengesizliği için ağırlık hesaplar.
    WeightCol ile model eğitiminde kullanılır.

    Args:
        df: Label kolonu içeren DataFrame
        label_col: Label kolon adı

    Returns:
        dict: {label_value: weight} sözlüğü
    """
    label_counts = df.groupBy(label_col).count().collect()
    total = sum(row["count"] for row in label_counts)
    n_classes = len(label_counts)

    weights = {}
    for row in label_counts:
        label_val = row[label_col]
        count = row["count"]
        # Balanced formula: total / (n_classes * count)
        weight = total / (n_classes * count)
        weights[label_val] = weight

    print("⚖️  Sınıf ağırlıkları:")
    for label_val, weight in sorted(weights.items()):
        label_name = "Normal" if label_val == 0.0 else "Attack"
        print(f"   {label_name} (label={int(label_val)}): weight={weight:.4f}")

    return weights


def add_weight_column(df: DataFrame, weights: dict, label_col: str = "label") -> DataFrame:
    """
    DataFrame'e weightCol ekler.

    Args:
        df: DataFrame
        weights: compute_class_weights'ten gelen ağırlık sözlüğü
        label_col: Label kolon adı

    Returns:
        DataFrame: 'classWeight' kolonu eklenmiş DataFrame
    """
    weight_expr = F.when(F.col(label_col) == 0.0, F.lit(weights.get(0.0, 1.0)))
    for label_val, weight in weights.items():
        if label_val != 0.0:
            weight_expr = weight_expr.when(F.col(label_col) == label_val, F.lit(weight))
    weight_expr = weight_expr.otherwise(F.lit(1.0))

    return df.withColumn("classWeight", weight_expr)


# ─────────────────────────────────────────────
#  5. Metrik Hesaplama
# ─────────────────────────────────────────────

def evaluate_model(predictions: DataFrame) -> dict:
    """
    Model tahminleri üzerinde tüm zorunlu metrikleri hesaplar.

    Zorunlu metrikler (PDF kuralı):
    - AUC-ROC
    - Accuracy
    - F1-Score
    - Precision
    - Recall

    Args:
        predictions: Model.transform() çıktısı (prediction, rawPrediction, probability kolonları)

    Returns:
        dict: Metrik adı -> değer
    """
    metrics = {}

    # 1. AUC-ROC (Binary)
    try:
        binary_eval = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label",
            metricName="areaUnderROC",
        )
        metrics["auc_roc"] = binary_eval.evaluate(predictions)
    except Exception as e:
        print(f"   ⚠️ AUC-ROC hesaplanamadı: {e}")
        metrics["auc_roc"] = 0.0

    # 2. Accuracy
    mc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    mc_eval.setMetricName("accuracy")
    metrics["accuracy"] = mc_eval.evaluate(predictions)

    # 3. F1-Score
    mc_eval.setMetricName("f1")
    metrics["f1_score"] = mc_eval.evaluate(predictions)

    # 4. Precision (Weighted)
    mc_eval.setMetricName("weightedPrecision")
    metrics["precision"] = mc_eval.evaluate(predictions)

    # 5. Recall (Weighted)
    mc_eval.setMetricName("weightedRecall")
    metrics["recall"] = mc_eval.evaluate(predictions)

    # Sonuçları yazdır
    print("\n📊 Model Değerlendirme Sonuçları:")
    print(f"   {'Metrik':<20} {'Değer':>10}")
    print(f"   {'─'*30}")
    for name, value in metrics.items():
        print(f"   {name:<20} {value:>10.4f}")

    return metrics


# ─────────────────────────────────────────────
#  6. Confusion Matrix
# ─────────────────────────────────────────────

def compute_confusion_matrix(predictions: DataFrame) -> dict:
    """
    Confusion matrix hesaplar ve detaylı rapor verir.

    Args:
        predictions: Model tahminleri

    Returns:
        dict: TP, TN, FP, FN değerleri
    """
    # prediction ve label üzerinden confusion matrix
    cm_df = (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )

    # Değerleri çıkar
    cm_data = {(int(row["label"]), int(row["prediction"])): row["count"] for row in cm_df.collect()}

    tn = cm_data.get((0, 0), 0)  # True Negative
    fp = cm_data.get((0, 1), 0)  # False Positive
    fn = cm_data.get((1, 0), 0)  # False Negative
    tp = cm_data.get((1, 1), 0)  # True Positive

    total = tn + fp + fn + tp

    result = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    print("\n🔲 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Normal  Attack")
    print(f"   Actual Normal   {tn:>6}  {fp:>6}")
    print(f"   Actual Attack   {fn:>6}  {tp:>6}")
    print(f"\n   Total: {total:,}")

    if total > 0:
        print(f"   True Positive Rate  (Recall):    {tp/(tp+fn)*100:.2f}%" if (tp+fn) > 0 else "")
        print(f"   True Negative Rate  (Specificity): {tn/(tn+fp)*100:.2f}%" if (tn+fp) > 0 else "")
        print(f"   False Positive Rate:  {fp/(fp+tn)*100:.2f}%" if (fp+tn) > 0 else "")
        print(f"   False Negative Rate:  {fn/(fn+tp)*100:.2f}%" if (fn+tp) > 0 else "")

    return result


# ─────────────────────────────────────────────
#  7. MLflow Logging
# ─────────────────────────────────────────────

def log_to_mlflow(
    run_name: str,
    model_type: str,
    params: dict,
    metrics: dict,
    model=None,
    confusion_matrix: dict = None,
    feature_importance: list = None,
    tags: dict = None,
) -> str:
    """
    Model sonuçlarını MLflow'a loglar.

    Args:
        run_name: MLflow run adı (örn: "logistic_regression_v1")
        model_type: Model tipi etiketi (örn: "LogisticRegression")
        params: Model parametreleri
        metrics: evaluate_model'den gelen metrikler
        model: PipelineModel veya CrossValidatorModel (opsiyonel)
        confusion_matrix: compute_confusion_matrix'ten gelen dict (opsiyonel)
        feature_importance: [(feature_name, importance)] listesi (opsiyonel)
        tags: Ek etiketler (opsiyonel)

    Returns:
        str: MLflow run ID
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Tag'ler
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("pipeline_stage", "training")
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, str(value))

        # Parametreler
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Metrikler
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Confusion Matrix metrikleri
        if confusion_matrix:
            for key, value in confusion_matrix.items():
                mlflow.log_metric(f"cm_{key}", value)

        # Model kaydet
        if model:
            try:
                mlflow.spark.log_model(model, artifact_path="spark-model")
                print(f"   ✅ Model MLflow'a kaydedildi.")
            except Exception as e:
                print(f"   ⚠️ Model kaydedilemedi: {e}")

        # Feature importance
        if feature_importance:
            fi_text = "Feature,Importance\n"
            for fname, fimportance in feature_importance:
                fi_text += f"{fname},{fimportance}\n"
            mlflow.log_text(fi_text, "feature_importance.csv")

        run_id = run.info.run_id
        print(f"\n✅ MLflow logging tamamlandı!")
        print(f"   Run ID:   {run_id}")
        print(f"   Run Name: {run_name}")
        print(f"   UI:       http://localhost:5000")

        return run_id


# ─────────────────────────────────────────────
#  8. Genel Pipeline Çalıştırıcı
# ─────────────────────────────────────────────

def run_ml_pipeline(
    spark: SparkSession,
    sample_size: int = None,
    split_log_stats: bool = True,
) -> tuple:
    """
    Gold veriden başlayarak ML pipeline'ın ortak adımlarını çalıştırır:
    1. Veri yükleme
    2. Feature hazırlığı
    3. Sınıf ağırlıkları
    4. Train/test split

    Tüm model script'leri bu fonksiyonu çağırarak başlar.

    Args:
        spark: SparkSession
        sample_size: Eğer verilirse, feature hazırlığı öncesi bu kadar satıra sınırlar
        split_log_stats: split sırasında ağır count/groupBy loglarını aç/kapat

    Returns:
        (train_df, test_df, feature_cols): Hazır eğitim ve test setleri + feature listesi
    """
    pipeline_start = time.perf_counter()
    print("\n🧱 Ortak ML pipeline başlatıldı...", flush=True)

    stage_start = time.perf_counter()
    init_mlflow()
    print(f"   ⏱️ MLflow: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    df = load_gold_data(spark, log_stats=False)
    print(f"   ⏱️ Gold okuma (lazy): {time.perf_counter() - stage_start:.2f}s", flush=True)

    feature_cols = get_feature_columns(df)

    stage_start = time.perf_counter()
    print(f"   ⏳ [1] prepare_features (lazy)...", flush=True)
    prepared_df = prepare_features(df, feature_cols=feature_cols, log_stats=False)
    print(f"   ✅ [1] prepare_features tanımlandı: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [2] Tek seferlik MATERIALIZE (coalesce + persist + count)...", flush=True)
    target_partitions = 4 if sample_size and sample_size <= 50000 else 8
    if sample_size is not None:
        total_count = df.count()
        fraction = min(sample_size / total_count, 1.0)
        print(f"   ⚡ Örneklem: {total_count:,} → ~{sample_size:,} satır (frac={fraction:.4f})", flush=True)
        prepared_df = prepared_df.sample(withReplacement=False, fraction=fraction, seed=42)

    prepared_df = prepared_df.coalesce(target_partitions).persist(StorageLevel.MEMORY_AND_DISK)
    materialized_count = prepared_df.count()
    print(f"   ✅ [2] Materialize tamam: {materialized_count:,} satır, {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [3] Class weights (cached'ten)...", flush=True)
    weights = compute_class_weights(prepared_df)
    prepared_df = add_weight_column(prepared_df, weights)
    print(f"   ✅ [3] Weights: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [4] randomSplit (cached'ten)...", flush=True)
    train_df, test_df = split_data(prepared_df, log_stats=False)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"   ✅ [4] Train: {train_count:,} | Test: {test_count:,} ({time.perf_counter() - stage_start:.2f}s)", flush=True)

    prepared_df.unpersist()

    print(f"\n🚀 ML Pipeline hazır! Toplam: {time.perf_counter() - pipeline_start:.2f}s", flush=True)
    return train_df, test_df, feature_cols


# ─────────────────────────────────────────────
#  9. MULTI-CLASS (Attack_type) Yardımcıları
# ─────────────────────────────────────────────
#
# Aşağıdaki fonksiyonlar binary versiyonların yanına eklenmiştir.
# Hedef: Attack_type kolonunu (string, çok-sınıflı) etiket olarak kullanan
# multinomial sınıflandırma pipeline'ı için ortak araçlar sağlamak.

def find_attack_type_column(df: DataFrame) -> str:
    """
    Gold tablodaki saldırı tipi kolonunu bulur.
    CSV/Silver akışında kolon adı 'Attack_type' veya 'attack_type' olarak
    farklılaşabildiği için iki olasılığı da kontrol eder.
    """
    candidates = ["Attack_type", "attack_type"]
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        "Gold tabloda Attack_type / attack_type kolonu bulunamadı. "
        f"Mevcut kolonlar: {df.columns}"
    )


def prepare_features_multiclass(
    df: DataFrame,
    label_col: str = None,
    feature_cols: list = None,
    log_stats: bool = False,
) -> tuple:
    """
    Multi-class sınıflandırma için feature ve etiket hazırlığı.

    Adımlar:
    1. Sayısal feature kolonlarını seçer (binary versiyon ile aynı kural).
    2. String tipindeki Attack_type kolonunu StringIndexer ile sayısal
       'label' kolonuna çevirir (en yoğun sınıf 0 olur).
    3. VectorAssembler ile 'features' vektörünü üretir.

    Returns:
        (assembled_df, label_index_model, feature_cols)
            assembled_df: 'features' ve 'label' kolonlarına sahip DataFrame
            label_index_model: StringIndexerModel (label index → string isim eşlemesi için)
            feature_cols: kullanılan numerik feature kolon listesi
    """
    if label_col is None:
        label_col = find_attack_type_column(df)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    print(
        f"🔧 Multi-class feature hazırlığı: "
        f"{len(feature_cols)} feature, label='{label_col}'"
    )

    indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep",
        stringOrderType="frequencyDesc",
    )
    label_index_model = indexer.fit(df.select(label_col))

    select_exprs = [
        F.coalesce(
            F.when(F.isnan(F.col(c).cast("double")), F.lit(0.0))
             .otherwise(F.col(c).cast("double")),
            F.lit(0.0),
        ).alias(c)
        for c in feature_cols
    ]
    select_exprs.append(F.col(label_col))
    df_typed = df.select(*select_exprs)

    indexed_df = label_index_model.transform(df_typed)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    assembled_df = assembler.transform(indexed_df).select("features", "label")
    assembled_df = assembled_df.filter(F.col("label").isNotNull())

    print(f"   ✅ Multi-class hazırlık tamam (sınıf sayısı: {len(label_index_model.labels)}).")
    print("   📚 Sınıf eşlemesi (label index → orijinal isim):")
    for idx, name in enumerate(label_index_model.labels):
        print(f"      {idx:>2} → {name}")

    if log_stats:
        final_count = assembled_df.count()
        print(f"   📊 Feature sonrası satır sayısı: {final_count:,}")

    return assembled_df, label_index_model, feature_cols


def compute_class_weights_multiclass(df: DataFrame, label_col: str = "label") -> dict:
    """
    N sınıflı dengesizlik için ağırlık hesaplar (sklearn 'balanced' formülü):
        weight = total / (n_classes * count_of_class)
    """
    label_counts = df.groupBy(label_col).count().collect()
    total = sum(row["count"] for row in label_counts)
    n_classes = len(label_counts)

    weights = {}
    for row in label_counts:
        label_val = float(row[label_col])
        count = row["count"]
        weights[label_val] = total / (n_classes * count) if count > 0 else 1.0

    print(f"⚖️  Multi-class sınıf ağırlıkları (n={n_classes}):")
    for label_val, weight in sorted(weights.items()):
        print(f"   label={int(label_val):>2}  weight={weight:.4f}")

    return weights


def add_weight_column_multiclass(
    df: DataFrame,
    weights: dict,
    label_col: str = "label",
) -> DataFrame:
    """
    Çoklu sınıflar için 'classWeight' kolonu ekler.
    F.create_map ile tek bir lookup yapılır (binary versiyondaki zincirleme
    when() yerine N sınıfta daha temiz çalışır).
    """
    if not weights:
        return df.withColumn("classWeight", F.lit(1.0))

    map_pairs = []
    for label_val, weight in weights.items():
        map_pairs.append(F.lit(float(label_val)))
        map_pairs.append(F.lit(float(weight)))

    weight_map = F.create_map(*map_pairs)
    return df.withColumn(
        "classWeight",
        F.coalesce(weight_map.getItem(F.col(label_col).cast("double")), F.lit(1.0)),
    )


def evaluate_model_multiclass(predictions: DataFrame, num_classes: int) -> dict:
    """
    Multi-class metrikleri hesaplar.

    Hesaplananlar:
        - accuracy
        - f1_score (weighted)
        - precision (weighted)
        - recall (weighted)
        - log_loss (varsa probability kolonu)
        - per-class precision/recall/f1 (label başına)

    Not: AUC-ROC binary'ye özgü olduğundan multi-class baz sette hesaplanmaz.
    İhtiyaç olursa One-vs-Rest ile ayrıca eklenebilir.
    """
    metrics = {}
    mc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )

    mc_eval.setMetricName("accuracy")
    metrics["accuracy"] = mc_eval.evaluate(predictions)

    mc_eval.setMetricName("f1")
    metrics["f1_score"] = mc_eval.evaluate(predictions)

    mc_eval.setMetricName("weightedPrecision")
    metrics["precision"] = mc_eval.evaluate(predictions)

    mc_eval.setMetricName("weightedRecall")
    metrics["recall"] = mc_eval.evaluate(predictions)

    try:
        mc_eval.setMetricName("logLoss")
        metrics["log_loss"] = mc_eval.evaluate(predictions)
    except Exception:
        pass

    print("\n📊 Multi-class Değerlendirme Sonuçları:")
    print(f"   {'Metrik':<20} {'Değer':>10}")
    print(f"   {'─'*30}")
    for name, value in metrics.items():
        print(f"   {name:<20} {value:>10.4f}")

    per_class_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    print("\n   Per-class metrikler:")
    print(f"   {'label':<6} {'precision':>10} {'recall':>10} {'f1':>10}")
    for c in range(num_classes):
        per_class_eval.setMetricLabel(float(c))
        try:
            per_class_eval.setMetricName("precisionByLabel")
            p = per_class_eval.evaluate(predictions)
            per_class_eval.setMetricName("recallByLabel")
            r = per_class_eval.evaluate(predictions)
            per_class_eval.setMetricName("fMeasureByLabel")
            f = per_class_eval.evaluate(predictions)
        except Exception:
            p, r, f = 0.0, 0.0, 0.0

        metrics[f"precision_class_{c}"] = p
        metrics[f"recall_class_{c}"] = r
        metrics[f"f1_class_{c}"] = f
        print(f"   {c:<6} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    return metrics


def compute_confusion_matrix_multiclass(
    predictions: DataFrame,
    label_names: list,
) -> dict:
    """
    NxN confusion matrix üretir ve label isimleriyle yazdırır.

    Returns:
        dict:
            "matrix"        : list[list[int]]  — confusion matrisi (label_names sırasında)
            "labels"        : list[str]        — sınıf isimleri (index sırası)
            "row_totals"    : list[int]        — her gerçek sınıfın toplamı
            "per_class_acc" : list[float]      — per-class doğruluk (recall)
    """
    n = len(label_names)
    cm_rows = (
        predictions
        .groupBy("label", "prediction")
        .count()
        .collect()
    )

    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for row in cm_rows:
        label_idx = int(row["label"])
        pred_idx = int(row["prediction"])
        if 0 <= label_idx < n and 0 <= pred_idx < n:
            matrix[label_idx][pred_idx] = int(row["count"])

    row_totals = [sum(row) for row in matrix]
    per_class_acc = [
        (matrix[i][i] / row_totals[i]) if row_totals[i] > 0 else 0.0
        for i in range(n)
    ]

    name_width = max((len(name) for name in label_names), default=8)
    name_width = min(max(name_width, 8), 24)

    print("\n🔲 Multi-class Confusion Matrix:")
    header = " " * (name_width + 2) + "│ " + "  ".join(
        f"{name[:name_width]:>{name_width}}" for name in label_names
    )
    print(header)
    print("─" * len(header))
    for i, name in enumerate(label_names):
        row_cells = "  ".join(f"{matrix[i][j]:>{name_width}}" for j in range(n))
        print(f" {name[:name_width]:<{name_width}} │ {row_cells}   (total={row_totals[i]:,}, acc={per_class_acc[i]*100:.1f}%)")

    return {
        "matrix": matrix,
        "labels": list(label_names),
        "row_totals": row_totals,
        "per_class_acc": per_class_acc,
    }


def run_ml_pipeline_multiclass(
    spark: SparkSession,
    sample_size: int = None,
    split_log_stats: bool = True,
) -> tuple:
    """
    Multi-class versiyonu için ortak ML pipeline.
    Binary `run_ml_pipeline` ile aynı iskelet, fakat label kolonu Attack_type
    (StringIndexer ile encode edilir).

    Returns:
        (train_df, test_df, feature_cols, label_index_model)
    """
    pipeline_start = time.perf_counter()
    print("\n🧱 Multi-class ML pipeline başlatıldı...", flush=True)

    stage_start = time.perf_counter()
    init_mlflow()
    print(f"   ⏱️ MLflow: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    df = load_gold_data(spark, log_stats=False)
    print(f"   ⏱️ Gold okuma (lazy): {time.perf_counter() - stage_start:.2f}s", flush=True)

    feature_cols = get_feature_columns(df)
    label_col = find_attack_type_column(df)

    stage_start = time.perf_counter()
    print(f"   ⏳ [1] prepare_features_multiclass (lazy)...", flush=True)
    prepared_df, label_index_model, feature_cols = prepare_features_multiclass(
        df, label_col=label_col, feature_cols=feature_cols, log_stats=False
    )
    print(f"   ✅ [1] prepare_features_multiclass: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [2] Tek seferlik MATERIALIZE (coalesce + persist + count)...", flush=True)
    target_partitions = 4 if sample_size and sample_size <= 50000 else 8
    if sample_size is not None:
        total_count = df.count()
        fraction = min(sample_size / total_count, 1.0)
        print(f"   ⚡ Örneklem: {total_count:,} → ~{sample_size:,} satır (frac={fraction:.4f})", flush=True)
        prepared_df = prepared_df.sample(withReplacement=False, fraction=fraction, seed=42)

    prepared_df = prepared_df.coalesce(target_partitions).persist(StorageLevel.MEMORY_AND_DISK)
    materialized_count = prepared_df.count()
    print(f"   ✅ [2] Materialize tamam: {materialized_count:,} satır, {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [3] Multi-class class weights...", flush=True)
    weights = compute_class_weights_multiclass(prepared_df)
    prepared_df = add_weight_column_multiclass(prepared_df, weights)
    print(f"   ✅ [3] Weights: {time.perf_counter() - stage_start:.2f}s", flush=True)

    stage_start = time.perf_counter()
    print(f"   ⏳ [4] randomSplit (cached'ten)...", flush=True)
    train_df, test_df = split_data(prepared_df, log_stats=False, stratified=True)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"   ✅ [4] Train: {train_count:,} | Test: {test_count:,} ({time.perf_counter() - stage_start:.2f}s)", flush=True)

    if split_log_stats:
        print("\n   Sınıf dağılımı (train+test):")
        dist_rows = (
            prepared_df.groupBy("label").count().orderBy("label").collect()
        )
        labels = label_index_model.labels
        for row in dist_rows:
            idx = int(row["label"])
            name = labels[idx] if 0 <= idx < len(labels) else f"label_{idx}"
            print(f"   {idx:>2}  {name:<25} {row['count']:>12,}")

    prepared_df.unpersist()

    print(f"\n🚀 Multi-class ML Pipeline hazır! Toplam: {time.perf_counter() - pipeline_start:.2f}s", flush=True)
    return train_df, test_df, feature_cols, label_index_model
