"""
Dashboard API — Kafka consumer + Delta Lake file-system stats.
SSE endpoint (/api/stream) her saniye tarayıcıya veri iter.
"""
import glob
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Delta yolları ──────────────────────────────────────────────────────────────
DELTA_ROOT = "/app/delta-storage"
LAYER_PATHS = {
    "Bronze": DELTA_ROOT + "/bronze/network_traffic",
    "Silver": DELTA_ROOT + "/silver/network_traffic",
    "Gold":   DELTA_ROOT + "/gold/ml_ready",
}

# ── Kafka consumer state ───────────────────────────────────────────────────────
stats = {
    "kafka_count": 0,
    "topic_total": 0,
    "msg_rate":    0,
    "kafka_connected": False,
    "attack_counts": {},
    "recent_messages": deque(maxlen=50),
    "started_at": time.time(),
}
_lock = threading.Lock()
_rate_window = deque(maxlen=60)


def kafka_consumer_thread():
    from kafka import KafkaConsumer
    while True:
        try:
            consumer = KafkaConsumer(
                "iot-network-traffic",
                bootstrap_servers="kafka:9092",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="dashboard-consumer-v3",
                enable_auto_commit=True,
            )
            print("[API] Kafka bağlantısı kuruldu.", flush=True)
            with _lock:
                stats["kafka_connected"] = True
            from kafka import TopicPartition
            batch, last_check, offset_check = 0, time.time(), 0

            # İlk bağlantıda hemen gerçek toplam mesaj sayısını al
            def _update_topic_total():
                try:
                    parts = consumer.partitions_for_topic("iot-network-traffic")
                    if parts:
                        tps = [TopicPartition("iot-network-traffic", p) for p in parts]
                        begins = consumer.beginning_offsets(tps)
                        ends = consumer.end_offsets(tps)
                        total = sum(ends[t] - begins[t] for t in tps)
                        with _lock:
                            stats["topic_total"] = total
                except Exception:
                    pass

            _update_topic_total()  # İlk ölçüm — bekleme yok

            while True:
                for tp, msgs in consumer.poll(timeout_ms=500).items():
                    for msg in msgs:
                        d = msg.value
                        with _lock:
                            stats["kafka_count"] += 1
                            attack = d.get("attack_type", d.get("Attack_type", "Normal"))
                            stats["attack_counts"][attack] = stats["attack_counts"].get(attack, 0) + 1
                            stats["recent_messages"].appendleft({
                                "time":    d.get("timestamp", ""),
                                "src":     d.get("source_ip", "?"),
                                "dst":     d.get("dest_ip", "?"),
                                "attack":  attack,
                                "flow_id": d.get("flow_id", ""),
                            })
                        batch += 1
                now = time.time()
                if now - last_check >= 1.0:
                    with _lock:
                        _rate_window.append(batch)
                        stats["msg_rate"] = sum(_rate_window) // max(len(_rate_window), 1)
                    batch, last_check = 0, now
                # Her 5s gerçek toplam mesaj sayısını güncelle
                if now - offset_check >= 5.0:
                    _update_topic_total()
                    offset_check = now
        except Exception as exc:
            with _lock:
                stats["kafka_connected"] = False
            print(f"[API] Kafka hatası: {exc}. 5s sonra tekrar.", flush=True)
            time.sleep(5)


# ── Layer row-count cache (her 30s güncellenir) ────────────────────────────────
_row_counts: dict = {"Bronze": 0, "Silver": 0, "Gold": 0}
_row_lock = threading.Lock()


def _count_parquet_rows(path: str) -> int:
    """Parquet metadata'dan satır sayısı okur — veri yüklemez, hızlıdır."""
    try:
        import pyarrow.parquet as pq
        pqs = glob.glob(path + "/**/*.parquet", recursive=True)
        return sum(pq.read_metadata(p).num_rows for p in pqs)
    except Exception:
        return 0


def _row_count_thread():
    while True:
        for name, path in LAYER_PATHS.items():
            if os.path.exists(path):
                count = _count_parquet_rows(path)
                with _row_lock:
                    _row_counts[name] = count
        time.sleep(10)


# ── Layer freshness ────────────────────────────────────────────────────────────
def _get_freshness() -> dict:
    result = {}
    now = time.time()
    for name, path in LAYER_PATHS.items():
        if not os.path.exists(path):
            result[name] = {"exists": False, "active": False, "has_data": False,
                            "rows": 0, "age_sec": None, "last_mod": "—"}
            continue
        try:
            mtimes = [
                os.path.getmtime(os.path.join(r, f))
                for r, _, files in os.walk(path)
                for f in files
            ]
            latest = max(mtimes) if mtimes else 0.0
        except Exception:
            latest = 0.0

        has_pq = bool(glob.glob(path + "/**/*.parquet", recursive=True))
        age = round(now - latest) if latest > 0 else None
        with _row_lock:
            rows = _row_counts.get(name, 0)
        result[name] = {
            "exists":   True,
            "active":   age is not None and age < 120,
            "has_data": has_pq,
            "rows":     rows,
            "age_sec":  age,
            "last_mod": datetime.fromtimestamp(latest).strftime("%H:%M:%S") if latest > 0 else "—",
        }
    return result


# ── SSE endpoint ───────────────────────────────────────────────────────────────
@app.route("/api/stream")
def stream_sse():
    def gen():
        while True:
            try:
                with _lock:
                    connected = stats["kafka_connected"]
                    kafka = {
                        "status": "ok" if connected else "waiting",
                        "total":  stats["topic_total"] or stats["kafka_count"],
                        "rate":   stats["msg_rate"],
                        "connected": connected,
                    }
                payload = json.dumps({
                    "ts":     datetime.now().isoformat(),
                    "kafka":  kafka,
                    "layers": _get_freshness(),
                })
                yield f"data: {payload}\n\n"
            except GeneratorExit:
                return
            except Exception:
                pass
            time.sleep(1)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ── Mevcut endpoint'ler ────────────────────────────────────────────────────────
@app.route("/api/stats")
def get_stats():
    with _lock:
        total_attacks = sum(v for k, v in stats["attack_counts"].items() if k != "Normal")
        total = stats["kafka_count"] or 1
        return jsonify({
            "kafka_count":    stats["kafka_count"],
            "msg_rate":       stats["msg_rate"],
            "attack_pct":     round((total_attacks / total) * 100, 1),
            "attack_counts":  dict(stats["attack_counts"]),
            "recent_messages": list(stats["recent_messages"]),
            "uptime":         int(time.time() - stats["started_at"]),
            "layer_rows":     dict(_row_counts),
        })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "kafka_count": stats["kafka_count"]})


@app.route("/api/layer/<layer_name>")
def get_layer_data(layer_name):
    paths = {"bronze": LAYER_PATHS["Bronze"], "silver": LAYER_PATHS["Silver"], "gold": LAYER_PATHS["Gold"]}
    if layer_name not in paths:
        return jsonify({"error": f"Unknown layer: {layer_name}"}), 400
    path = paths[layer_name]
    if not os.path.exists(path):
        return jsonify({"layer": layer_name, "status": "empty", "columns": [], "rows": [], "total_rows": 0})
    try:
        from deltalake import DeltaTable
        df = DeltaTable(path).to_pandas()
        total = len(df)
        sample = df.tail(20).iloc[::-1]
        cols = list(sample.columns)
        if len(cols) > 15:
            priority = ["timestamp", "source_ip", "dest_ip", "attack_type", "flow_id",
                        "Attack_label", "ingestion_time", "json_payload"]
            ordered = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
            sample = sample[ordered[:15]]
        sample = sample.fillna("null")
        return jsonify({"layer": layer_name, "status": "ok",
                        "columns": list(sample.columns),
                        "rows": sample.to_dict(orient="records"),
                        "total_rows": total})
    except Exception as exc:
        return jsonify({"layer": layer_name, "status": "error", "message": str(exc),
                        "columns": [], "rows": [], "total_rows": 0})


if __name__ == "__main__":
    for target in (kafka_consumer_thread, _row_count_thread):
        threading.Thread(target=target, daemon=True).start()
    print("[API] http://0.0.0.0:5001", flush=True)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
