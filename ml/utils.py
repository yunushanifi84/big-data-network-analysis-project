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
import mlflow
import mlflow.spark
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
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

    row_count = df.count()
    col_count = len(df.columns)
    print(f"   ✅ {row_count:,} satır, {col_count} kolon yüklendi.")

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

    # 1. Null/NaN değerleri temizle (VectorAssembler NaN kaldıramaz)
    for col_name in feature_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), F.lit(0.0))
            .when(F.isnan(F.col(col_name)), F.lit(0.0))
            .otherwise(F.col(col_name).cast("double"))
        )

    # 2. VectorAssembler — tüm feature'ları tek bir vektöre birleştir
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",  # Hatalı satırları atla
    )
    assembled_df = assembler.transform(df)

    # 3. Label kolonu — zaten 0/1 integer ise doğrudan cast et
    assembled_df = assembled_df.withColumn(
        "label", F.col(label_col).cast("double")
    )

    # Label null olanları filtrele
    assembled_df = assembled_df.filter(F.col("label").isNotNull())

    final_count = assembled_df.count()
    print(f"   ✅ Feature hazırlığı tamamlandı. {final_count:,} satır kullanıma hazır.")

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
    test_ratio = 1.0 - train_ratio

    if stratified:
        # Label bazlı stratified split:
        # Her sınıf için ayrı split yapıp birleştiririz, böylece sınıf oranı korunur.
        labels = [row["label"] for row in df.select("label").distinct().collect()]
        train_parts = []
        test_parts = []

        for label_value in labels:
            label_df = df.filter(F.col("label") == label_value)
            label_train, label_test = label_df.randomSplit([train_ratio, test_ratio], seed=seed)
            train_parts.append(label_train)
            test_parts.append(label_test)

        train_df = train_parts[0]
        test_df = test_parts[0]
        for part in train_parts[1:]:
            train_df = train_df.unionByName(part)
        for part in test_parts[1:]:
            test_df = test_df.unionByName(part)
    else:
        # randomSplit — Spark-native bölme
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
    total = df.count()
    label_counts = df.groupBy(label_col).count().collect()
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

def run_ml_pipeline(spark: SparkSession) -> tuple:
    """
    Gold veriden başlayarak ML pipeline'ın ortak adımlarını çalıştırır:
    1. Veri yükleme
    2. Feature hazırlığı
    3. Sınıf ağırlıkları
    4. Train/test split

    Tüm model script'leri bu fonksiyonu çağırarak başlar.

    Returns:
        (train_df, test_df, feature_cols): Hazır eğitim ve test setleri + feature listesi
    """
    # 1. MLflow bağlantısı
    init_mlflow()

    # 2. Gold katmanından veri yükle
    df = load_gold_data(spark)

    # 3. Feature kolonlarını belirle
    feature_cols = get_feature_columns(df)

    # 4. Feature vektörü ve label oluştur
    prepared_df = prepare_features(df, feature_cols=feature_cols)

    # 5. Sınıf ağırlıklarını hesapla ve ekle
    weights = compute_class_weights(prepared_df)
    prepared_df = add_weight_column(prepared_df, weights)

    # 6. Train/Test split
    train_df, test_df = split_data(prepared_df)

    # Cache — ML eğitimi sırasında tekrar tekrar okunacak
    train_df = train_df.cache()
    test_df = test_df.cache()

    print(f"\n🚀 ML Pipeline hazır! Model eğitimine geçilebilir.")

    return train_df, test_df, feature_cols
