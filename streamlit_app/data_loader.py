"""
Streamlit Dashboard — Veri Yükleyici
Tüm veri kaynaklarını (MLflow SQLite, Delta tabloları, feature importance CSV'leri)
tek bir cache'li API arkasında toplar. Streamlit @cache_data ile yeniden hesaplama
maliyetini sıfırlar.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ── Yol Sabitleri ────────────────────────────────────────────────────────────
# Hem Docker konteyneri (/app/...) hem de lokal çalıştırma için çalışsın.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MLFLOW_DB = REPO_ROOT / "mlruns" / "mlflow.db"
DEFAULT_DELTA_ROOT = REPO_ROOT / "delta-storage"
DEFAULT_ML_DIR = REPO_ROOT / "ml"
DEFAULT_RAW_DATA = REPO_ROOT / "data" / "raw"

MLFLOW_DB = Path(os.environ.get("MLFLOW_DB", DEFAULT_MLFLOW_DB))
DELTA_ROOT = Path(os.environ.get("DELTA_ROOT", DEFAULT_DELTA_ROOT))
ML_DIR = Path(os.environ.get("ML_DIR", DEFAULT_ML_DIR))
KAFKA_BROKERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092").split(",")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "iot-network-traffic")


# ── Tüm 5 Model için Ortak Tanım ─────────────────────────────────────────────
# MLflow'a loglanmamış modeller için fallback metrik kaynağı görevi görür.
# Gerçek metrik MLflow'da varsa onunla üzerine yazılır.
MODELS: List[Dict] = [
    {
        "key": "logistic_regression",
        "display": "Logistic Regression",
        "fi_csv": None,  # LR coefficients ayrı; placeholder
        "color": "#6366F1",
        "mlflow_pattern": "logistic_regression%",
    },
    {
        "key": "decision_tree",
        "display": "Decision Tree",
        "fi_csv": "dt_feature_importance.csv",
        "color": "#10B981",
        "mlflow_pattern": "decision_tree%",
    },
    {
        "key": "random_forest",
        "display": "Random Forest",
        "fi_csv": "rf_feature_importance.csv",
        "color": "#F59E0B",
        "mlflow_pattern": "random_forest%",
    },
    {
        "key": "gbt",
        "display": "Gradient Boosted Trees",
        "fi_csv": "gbt_feature_importance.csv",
        "color": "#EF4444",
        "mlflow_pattern": "gbt%",
    },
    {
        "key": "naive_bayes",
        "display": "Naive Bayes",
        "fi_csv": "nb_feature_importance.csv",
        "color": "#8B5CF6",
        "mlflow_pattern": "naive_bayes%",
    },
]

MODEL_BY_KEY = {m["key"]: m for m in MODELS}


# ── MLflow Okuyucusu ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_mlflow_runs() -> pd.DataFrame:
    """MLflow SQLite'ından tüm FINISHED run'ları metrik+tag birleşimi ile döner."""
    if not MLFLOW_DB.exists():
        return pd.DataFrame()

    con = sqlite3.connect(str(MLFLOW_DB))
    try:
        runs = pd.read_sql_query(
            """
            SELECT run_uuid, name, status, start_time, end_time, experiment_id
            FROM runs
            WHERE status = 'FINISHED'
            ORDER BY start_time DESC
            """,
            con,
        )
        if runs.empty:
            return runs

        tags = pd.read_sql_query(
            """
            SELECT run_uuid, key, value
            FROM tags
            WHERE key IN ('model_type', 'mlflow.runName', 'num_classes',
                          'classification_type', 'stage')
            """,
            con,
        )
        metrics = pd.read_sql_query(
            "SELECT run_uuid, key, value FROM metrics", con
        )
        params = pd.read_sql_query(
            "SELECT run_uuid, key, value FROM params", con
        )
    finally:
        con.close()

    # Tag'leri pivotla
    tag_pivot = (
        tags.pivot_table(index="run_uuid", columns="key", values="value", aggfunc="first")
        .reset_index()
    )
    # Metrikleri pivotla (latest değer)
    metric_pivot = (
        metrics.groupby(["run_uuid", "key"], as_index=False)
        .last()
        .pivot(index="run_uuid", columns="key", values="value")
        .reset_index()
    )

    df = runs.merge(tag_pivot, on="run_uuid", how="left")
    df = df.merge(metric_pivot, on="run_uuid", how="left")

    # Zaman damgaları okunabilir
    df["start_dt"] = pd.to_datetime(df["start_time"], unit="ms", errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_time"], unit="ms", errors="coerce")
    df["duration_sec"] = ((df["end_time"] - df["start_time"]) / 1000).round(1)

    return df


def _normalize_model_type(value: Optional[str]) -> Optional[str]:
    """
    MLflow 'model_type' tag'ini veya run name'ini bizim 'key' alanımıza eşler.
    Boşluk/alt çizgi/tire farklılıklarına dayanıklı.

    NOT: Artık tüm modeller multi-class (Attack_type) için eğitiliyor.
    Eski "multiclass_v1" isimli LR run'ı da geriye dönük olarak logistic_regression'a
    eşlenir; "OneVsRest" suffix'li olan GBT olduğu için gbt'ye eşlenir.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.lower().replace(" ", "").replace("_", "").replace("-", "")
    if "logistic" in v:
        return "logistic_regression"
    if "decisiontree" in v or v.startswith("dt"):
        return "decision_tree"
    if "randomforest" in v or v.startswith("rf"):
        return "random_forest"
    if "gradient" in v or "gbt" in v:
        return "gbt"
    if "naive" in v or "bayes" in v or v.startswith("nb"):
        return "naive_bayes"
    return None


@st.cache_data(ttl=30)
def get_best_run_per_model() -> pd.DataFrame:
    """
    Her model_type için en iyi run'ı döner.

    Strateji:
    1) Run'ları model anahtarına eşle (tag → fallback run name).
    2) Mevcut sınıflandırma tipini tercih et: 'multiclass' tag'i olan
       run'lar binary olanlardan ÖNCELİKLİDİR. (Proje artık multi-class.)
    3) Aynı sınıflandırma tipi içinde F1-Score'a göre sırala.
       (Multi-class'ta accuracy sınıf dengesizliğinden çok etkilenir, F1 daha dürüst.)
    """
    runs = load_mlflow_runs()
    rows: List[Dict] = []

    if not runs.empty:
        runs = runs.copy()
        runs["__model_key"] = runs.get("model_type", pd.Series([None] * len(runs))).map(
            _normalize_model_type
        )
        name_fallback = runs["name"].fillna("").str.lower().map(_normalize_model_type)
        runs["__model_key"] = runs["__model_key"].fillna(name_fallback)
        # multiclass tag'i olanlar 1, olmayanlar 0 — desc sıralamada multiclass öne geçer
        runs["__is_multiclass"] = (
            runs.get("classification_type", pd.Series([None] * len(runs)))
            .fillna("")
            .str.lower()
            .eq("multiclass")
            .astype(int)
        )

    for model in MODELS:
        key = model["key"]
        row = {
            "model_key": key,
            "display": model["display"],
            "color": model["color"],
            "run_id": None,
            "accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "auc_roc": None,
            "log_loss": None,
            "duration_sec": None,
            "num_classes": None,
            "classification_type": None,
            "has_mlflow": False,
        }
        if not runs.empty:
            sub = runs[runs["__model_key"] == key]
            if not sub.empty:
                # Önce multiclass'ı tercih et, sonra F1-Score (yoksa accuracy)
                sort_cols = ["__is_multiclass"]
                if "f1_score" in sub.columns:
                    sort_cols.append("f1_score")
                if "accuracy" in sub.columns:
                    sort_cols.append("accuracy")
                best = sub.sort_values(
                    sort_cols, ascending=[False] * len(sort_cols), na_position="last"
                ).iloc[0]
                row["run_id"] = best["run_uuid"]
                for m in ("accuracy", "f1_score", "precision", "recall", "auc_roc", "log_loss"):
                    if m in sub.columns:
                        row[m] = float(best[m]) if pd.notna(best[m]) else None
                row["duration_sec"] = (
                    float(best["duration_sec"]) if pd.notna(best.get("duration_sec")) else None
                )
                row["num_classes"] = best.get("num_classes")
                row["classification_type"] = best.get("classification_type")
                row["has_mlflow"] = True
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def get_run_metrics_full(run_id: str) -> Dict[str, float]:
    """Tek bir run'ın tüm metriklerini dict olarak döner."""
    if not MLFLOW_DB.exists() or not run_id:
        return {}
    con = sqlite3.connect(str(MLFLOW_DB))
    try:
        df = pd.read_sql_query(
            "SELECT key, value FROM metrics WHERE run_uuid = ?",
            con,
            params=(run_id,),
        )
    finally:
        con.close()
    if df.empty:
        return {}
    return dict(zip(df["key"], df["value"]))


# ── Feature Importance CSV'leri ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_feature_importance(model_key: str, top_n: int = 15) -> pd.DataFrame:
    model = MODEL_BY_KEY.get(model_key)
    if not model or not model.get("fi_csv"):
        return pd.DataFrame()
    path = ML_DIR / model["fi_csv"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# ── Delta Lake Katmanları ────────────────────────────────────────────────────
def _try_read_delta(path: Path) -> Optional[pd.DataFrame]:
    """delta-rs ile dene; yoksa parquet parçalarını oku."""
    if not path.exists():
        return None
    # Önce delta-rs
    try:
        from deltalake import DeltaTable

        dt = DeltaTable(str(path))
        return dt.to_pandas()
    except Exception:
        pass
    # Fallback: parquet glob (Delta log yok sayılır — son snapshot ≈ tüm parquet'ler)
    try:
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            return None
        return pd.concat((pd.read_parquet(p) for p in parquet_files), ignore_index=True)
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_layer_stats() -> pd.DataFrame:
    """Bronze/Silver/Gold katmanları için satır sayısı, kolon sayısı, boyut bilgisi."""
    layers = [
        ("Bronze", DELTA_ROOT / "bronze" / "network_traffic", "Kafka ham JSON"),
        ("Silver", DELTA_ROOT / "silver" / "network_traffic", "Parse + temizlik"),
        ("Gold", DELTA_ROOT / "gold" / "ml_ready", "ML hazır + 5 yeni feature"),
    ]
    rows = []
    for name, path, desc in layers:
        size_mb = 0.0
        rowcount = None
        cols = None
        if path.exists():
            # Boyut: tüm parquet ve _delta_log
            for p in path.rglob("*"):
                if p.is_file():
                    size_mb += p.stat().st_size / (1024 * 1024)
            df = _try_read_delta(path)
            if df is not None:
                rowcount = len(df)
                cols = len(df.columns)
        rows.append(
            {
                "layer": name,
                "description": desc,
                "rows": rowcount,
                "columns": cols,
                "size_mb": round(size_mb, 2),
                "exists": path.exists(),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=8)
def get_kafka_topic_offsets() -> dict:
    """Kafka topic'teki toplam mesaj sayısı ve broker bağlantı durumu."""
    try:
        from kafka import KafkaConsumer, TopicPartition  # noqa: PLC0415

        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BROKERS,
            consumer_timeout_ms=3000,
            request_timeout_ms=5000,
            connections_max_idle_ms=6000,
        )
        partitions_set = consumer.partitions_for_topic(KAFKA_TOPIC)
        if not partitions_set:
            consumer.close()
            return {"status": "no_topic", "total_messages": 0, "partitions": 0, "topic": KAFKA_TOPIC}

        tps = [TopicPartition(KAFKA_TOPIC, p) for p in partitions_set]
        end_offsets = consumer.end_offsets(tps)
        consumer.close()

        total = sum(end_offsets[tp] for tp in tps)
        return {
            "status": "ok",
            "total_messages": total,
            "partitions": len(partitions_set),
            "topic": KAFKA_TOPIC,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120], "total_messages": 0, "partitions": 0}


@st.cache_data(ttl=5)
def get_layer_freshness() -> Dict[str, dict]:
    """Bronze/Silver/Gold katmanlarının son değişiklik zamanını döner."""
    result: Dict[str, dict] = {}
    now = time.time()
    layers = {
        "Bronze": DELTA_ROOT / "bronze" / "network_traffic",
        "Silver": DELTA_ROOT / "silver" / "network_traffic",
        "Gold":   DELTA_ROOT / "gold" / "ml_ready",
    }
    for name, path in layers.items():
        if not path.exists():
            result[name] = {"active": False, "last_modified": None, "age_sec": None, "exists": False}
            continue
        latest_mtime = max(
            (p.stat().st_mtime for p in path.rglob("*") if p.is_file()),
            default=0.0,
        )
        if latest_mtime == 0.0:
            result[name] = {"active": False, "last_modified": None, "age_sec": None, "exists": True}
            continue
        age_sec = now - latest_mtime
        result[name] = {
            "active": age_sec < 120,
            "last_modified": datetime.fromtimestamp(latest_mtime).strftime("%H:%M:%S"),
            "age_sec": round(age_sec),
            "exists": True,
        }
    return result


@st.cache_data(ttl=60)
def load_gold_sample(limit: int = 0) -> pd.DataFrame:
    """Gold tablosundan veri. limit>0 ise örnekler, yoksa tüm tablo."""
    df = _try_read_delta(DELTA_ROOT / "gold" / "ml_ready_compact")
    if df is None:
        df = _try_read_delta(DELTA_ROOT / "gold" / "ml_ready")
    if df is None:
        return pd.DataFrame()
    if limit > 0 and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def load_silver_sample(limit: int = 30_000) -> pd.DataFrame:
    df = _try_read_delta(DELTA_ROOT / "silver" / "network_traffic")
    if df is None:
        return pd.DataFrame()
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df


# ── Türetilmiş Özellik Metaverisi ────────────────────────────────────────────
ENGINEERED_FEATURES = [
    {
        "name": "traffic_asymmetry_ratio",
        "title": "Trafik Asimetri Oranı",
        "formula": "tcp_ack / (tcp_seq + 1)",
        "why": (
            "Normal trafik simetrik ACK/SEQ akışına sahiptir. DDoS / SYN-Flood "
            "saldırılarında saldırgan çok sayıda SYN gönderir ama ACK dönmez — "
            "bu oran sapar."
        ),
        "detects": ["DDoS TCP/UDP/ICMP Flood", "SYN Flood"],
        "icon": "🔀",
    },
    {
        "name": "pkt_size_cv",
        "title": "Paket Boyutu Varyasyon Katsayısı",
        "formula": "|tcp_len − tcp_payload| / (tcp_len + 1)",
        "why": (
            "Normal trafikte paket boyutları tutarlıdır. Port Scanning ve "
            "Vulnerability Scanner saldırıları farklı boyutlarda paketler "
            "üreterek varyasyonu yükseltir."
        ),
        "detects": ["Port Scanning", "Vulnerability Scanner"],
        "icon": "📦",
    },
    {
        "name": "flow_intensity",
        "title": "Akış Yoğunluğu",
        "formula": "tcp_len / (frame_time_relative + ε)",
        "why": (
            "Birim zamandaki veri yoğunluğu. Flooding saldırıları çok kısa sürede "
            "yüksek hacimli trafik üreterek bu metriği patlatır."
        ),
        "detects": ["DDoS Flood", "Brute-Force"],
        "icon": "🌊",
    },
    {
        "name": "iat_regularity",
        "title": "Inter-Arrival Time Düzenliliği",
        "formula": "std(arrival_intervals) / (mean + ε)",
        "why": (
            "Otomatik saldırı araçları çok düzenli aralıklarla paket gönderir — "
            "insan/legitimate trafiğine göre çok düşük varyans gösterir."
        ),
        "detects": ["Botnet trafiği", "MITM", "Otomatize sömürü"],
        "icon": "⏱️",
    },
    {
        "name": "conn_efficiency",
        "title": "Bağlantı Verimliliği",
        "formula": "tcp_payload / (tcp_len + 1)",
        "why": (
            "Verimli bağlantılarda payload/header oranı yüksektir. Saldırılar "
            "çoğunlukla payload'sız veya minimal payload'lı kontrol paketleri kullanır."
        ),
        "detects": ["Reconnaissance", "ICMP Flood", "Null packet attacks"],
        "icon": "⚡",
    },
]


# ── Görselleştirme Yardımcıları ──────────────────────────────────────────────
PALETTE = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#64748B",
    "bg": "#0F172A",
    "panel": "#1E293B",
    "text": "#E2E8F0",
}

ATTACK_COLORS = [
    "#10B981", "#EF4444", "#F59E0B", "#6366F1", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#84CC16", "#06B6D4",
    "#A855F7", "#F43F5E", "#22C55E", "#3B82F6", "#EAB308",
]
