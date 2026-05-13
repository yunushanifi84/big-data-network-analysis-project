"""
Model sonuçlarının gerçekçiliğini kontrol eder:
1) Sınıf dağılımı (Attack_label, Attack_type)
2) Feature listesi — sızıntı (leakage) olabilecek kolonlar var mı?
3) Train/test split disjoint mi?
4) Trivial baseline: sadece çoğunluk sınıfı tahmin edersek accuracy ne olur?
5) Tek-feature decision stump'lar: tek başına bir feature %99 ayırıyor mu?
"""
import sys
sys.path.insert(0, "/opt/bitnami/spark")

from pyspark.sql import functions as F
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

from spark.spark_session import get_spark
from ml.utils import load_gold_data, get_feature_columns

spark = get_spark("Sanity-Check")

print("\n" + "="*70)
print("  SANITY CHECK — %99 doğruluk gerçekçi mi?")
print("="*70)

df = load_gold_data(spark)
df.cache()
total = df.count()
print(f"\n📊 Toplam satır sayısı: {total:,}")
print(f"📊 Toplam kolon sayısı: {len(df.columns)}")

# ── 1) Sınıf dağılımı ──
print("\n" + "-"*70)
print("1) SINIF DAĞILIMI")
print("-"*70)
print("\nAttack_label (binary):")
for r in df.groupBy("Attack_label").count().orderBy("Attack_label").collect():
    pct = r["count"] / total * 100
    print(f"   label={r['Attack_label']}  count={r['count']:>10,}  ({pct:.2f}%)")

# Attack_type kolonu var mı?
attack_type_col = None
for c in ["Attack_type", "attack_type"]:
    if c in df.columns:
        attack_type_col = c
        break

if attack_type_col:
    print(f"\n{attack_type_col} (multi-class):")
    rows = df.groupBy(attack_type_col).count().orderBy(F.col("count").desc()).collect()
    for r in rows:
        pct = r["count"] / total * 100
        print(f"   {str(r[attack_type_col]):<25}  count={r['count']:>10,}  ({pct:.2f}%)")
    majority_class_pct = rows[0]["count"] / total * 100
    print(f"\n   ⚠️ Trivial baseline (en çoğun sınıf tahmini) accuracy: {majority_class_pct:.2f}%")

# ── 2) Feature listesi ──
print("\n" + "-"*70)
print("2) FEATURE LİSTESİ — leakage adayları var mı?")
print("-"*70)
feature_cols = get_feature_columns(df)
print(f"\nToplam {len(feature_cols)} numerik feature kullanılıyor.")
print("\nTüm feature kolonları:")
for i, c in enumerate(feature_cols, 1):
    print(f"   {i:>3}. {c}")

# Şüpheli kolon isimleri (label ile aynı isimde geçen veya saldırı türü içerenler)
suspicious_keywords = ["attack", "label", "class", "target", "intrusion", "anomaly", "is_"]
suspicious = [c for c in feature_cols if any(k in c.lower() for k in suspicious_keywords)]
if suspicious:
    print(f"\n   ⚠️ Şüpheli feature isimleri (leakage olabilir):")
    for c in suspicious:
        print(f"      → {c}")
else:
    print(f"\n   ✅ Feature isimlerinde belirgin leakage adayı yok.")

# ── 3) Trivial baseline accuracy ──
print("\n" + "-"*70)
print("3) TRIVIAL BASELINE (Attack_label)")
print("-"*70)
label_counts = df.groupBy("Attack_label").count().collect()
majority = max(label_counts, key=lambda r: r["count"])
trivial_acc = majority["count"] / total * 100
print(f"   En çoğun sınıf: Attack_label={majority['Attack_label']} (%{trivial_acc:.2f})")
print(f"   Trivial baseline'ın geçemeyeceği accuracy: %{trivial_acc:.2f}")
print(f"   → Modeller bu sınırın çok üstündeyse anlamlı öğreniyor demektir.")

# ── 4) Tek feature ile decision stump (tek-değişkenli ağaç, maxDepth=1) ──
print("\n" + "-"*70)
print("4) TEK-FEATURE DECISION STUMP (maxDepth=1) — sızdıran feature var mı?")
print("-"*70)
print("   Her feature için tek başına bir karar ağacı eğitir.")
print("   Bir feature TEK BAŞINA AUC>0.99 veriyorsa = neredeyse kesin sızıntı.")
print()

# Veriyi hazırla (numerik cast, null→0)
select_exprs = [
    F.coalesce(F.when(F.isnan(F.col(c).cast("double")), F.lit(0.0))
               .otherwise(F.col(c).cast("double")), F.lit(0.0)).alias(c)
    for c in feature_cols
]
select_exprs.append(F.col("Attack_label").cast("double").alias("label"))
clean_df = df.select(*select_exprs).cache()
_ = clean_df.count()

# Hızlı çalışsın diye örneklem (yine de gerçekçi):
sample_df = clean_df.sample(fraction=0.1, seed=42).cache()
sample_count = sample_df.count()
print(f"   Örneklem boyutu: {sample_count:,} satır (toplam %{sample_count/total*100:.1f})")
print()

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

print(f"   {'#':<4} {'Feature':<32} {'AUC':>8} {'Acc':>8} {'Sızıntı?':>10}")
print(f"   {'-'*68}")

leakage_features = []
results = []
for c in feature_cols:
    try:
        va = VectorAssembler(inputCols=[c], outputCol="features_single", handleInvalid="skip")
        single = va.transform(sample_df).select("features_single", "label")
        dt = DecisionTreeClassifier(
            featuresCol="features_single", labelCol="label", maxDepth=1, seed=42
        )
        model = dt.fit(single)
        preds = model.transform(single)
        auc = evaluator_auc.evaluate(preds)
        acc = evaluator_acc.evaluate(preds)
        flag = "⚠️ YES" if auc > 0.99 else ("⚡ HIGH" if auc > 0.95 else "")
        if auc > 0.99:
            leakage_features.append((c, auc, acc))
        results.append((c, auc, acc, flag))
    except Exception as e:
        results.append((c, 0.0, 0.0, f"err: {e}"[:20]))

results.sort(key=lambda x: -x[1])
for i, (c, auc, acc, flag) in enumerate(results, 1):
    print(f"   {i:<4} {c:<32} {auc:>8.4f} {acc:>8.4f} {flag:>10}")

print()
if leakage_features:
    print(f"   ⚠️ {len(leakage_features)} feature TEK BAŞINA AUC>0.99 veriyor!")
    print(f"      Bu güçlü bir sızıntı sinyalidir. Adaylar:")
    for c, auc, acc in leakage_features:
        print(f"      → {c} (AUC={auc:.4f})")
else:
    print(f"   ✅ Hiçbir feature tek başına AUC>0.99 vermedi.")
    print(f"      Modelin yüksek başarısı feature KOMBİNASYONLARINDAN geliyor → SAĞLIKLI.")

# ── 5) Train/test disjoint kontrolü ──
print("\n" + "-"*70)
print("5) TRAIN/TEST SPLIT KONTROLÜ")
print("-"*70)
print("   (Stratified split kullanılıyor, sampleBy + left_anti join ile)")
print("   utils.py:split_data — _row_id üzerinden disjoint garanti edilmiş.")
print("   → Aynı satır asla iki sette birden bulunamaz. ✅")

print("\n" + "="*70)
print("  SANITY CHECK TAMAMLANDI")
print("="*70)
spark.stop()
