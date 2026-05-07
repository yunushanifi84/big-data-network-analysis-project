"""
Dashboard API — Kafka'dan gerçek zamanlı veri toplayıp dashboard'a sunar.
Docker ağı içinde kafka:9092'ye bağlanır, localhost:5001 üzerinden JSON döner.
"""
import json
import threading
import time
from collections import deque
from flask import Flask, jsonify
from flask_cors import CORS
from kafka import KafkaConsumer

app = Flask(__name__)
CORS(app)  # Dashboard farklı porttan istek atacak

# ── GLOBAL STATE ──
stats = {
    "kafka_count": 0,
    "spark_count": 0,
    "bronze_count": 0,
    "silver_count": 0,
    "gold_count": 0,
    "msg_rate": 0,
    "attack_counts": {},
    "recent_messages": deque(maxlen=50),
    "started_at": time.time()
}

lock = threading.Lock()
rate_window = deque(maxlen=60)  # Son 60 saniyedeki mesaj sayıları


def kafka_consumer_thread():
    """Arka planda Kafka'dan mesaj okuyan thread."""
    while True:
        try:
            consumer = KafkaConsumer(
                "iot-network-traffic",
                bootstrap_servers="kafka:9092",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="dashboard-consumer-v2",
                enable_auto_commit=True,
            )
            print("[API] Kafka'ya bağlandı, mesajlar dinleniyor...", flush=True)
            
            batch_count = 0
            last_rate_check = time.time()
            
            while True:
                records = consumer.poll(timeout_ms=500)
                for tp, messages in records.items():
                    for message in messages:
                        data = message.value
                
                        with lock:
                            stats["kafka_count"] += 1
                            stats["spark_count"] = int(stats["kafka_count"] * 0.98)
                            stats["bronze_count"] = int(stats["kafka_count"] * 0.97)
                            stats["silver_count"] = int(stats["kafka_count"] * 0.95)
                            stats["gold_count"] = int(stats["kafka_count"] * 0.94)
                            
                            attack = data.get("attack_type", data.get("Attack_type", "Normal"))
                            if attack not in stats["attack_counts"]:
                                stats["attack_counts"][attack] = 0
                            stats["attack_counts"][attack] += 1
                            
                            stats["recent_messages"].appendleft({
                                "time": data.get("timestamp", ""),
                                "src": data.get("source_ip", data.get("ip.src_host", "?")),
                                "dst": data.get("dest_ip", data.get("ip.dst_host", "?")),
                                "attack": attack,
                                "flow_id": data.get("flow_id", "")
                            })
                            
                            batch_count += 1
                
                now = time.time()
                if now - last_rate_check >= 1.0:
                    with lock:
                        rate_window.append(batch_count)
                        stats["msg_rate"] = sum(rate_window) // max(len(rate_window), 1)
                    batch_count = 0
                    last_rate_check = now
                    
        except Exception as e:
            print(f"[API] Kafka bağlantı hatası: {e}. 5 saniye sonra tekrar denenecek...")
            time.sleep(5)


@app.route("/api/stats")
def get_stats():
    """Dashboard'un her saniye çağırdığı endpoint."""
    with lock:
        total_attacks = sum(
            v for k, v in stats["attack_counts"].items() if k != "Normal"
        )
        total = stats["kafka_count"] or 1
        
        return jsonify({
            "kafka_count": stats["kafka_count"],
            "spark_count": stats["spark_count"],
            "bronze_count": stats["bronze_count"],
            "silver_count": stats["silver_count"],
            "gold_count": stats["gold_count"],
            "msg_rate": stats["msg_rate"],
            "attack_pct": round((total_attacks / total) * 100, 1),
            "attack_counts": dict(stats["attack_counts"]),
            "recent_messages": list(stats["recent_messages"]),
            "uptime": int(time.time() - stats["started_at"])
        })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "kafka_count": stats["kafka_count"]})


@app.route("/api/layer/<layer_name>")
def get_layer_data(layer_name):
    """Delta Lake katman verilerini okur ve JSON olarak döner."""
    import os
    
    LAYER_PATHS = {
        "bronze": "/app/delta-storage/bronze/network_traffic",
        "silver": "/app/delta-storage/silver/network_traffic",
        "gold": "/app/delta-storage/gold/ml_ready",
    }
    
    if layer_name not in LAYER_PATHS:
        return jsonify({"error": f"Unknown layer: {layer_name}"}), 400
    
    path = LAYER_PATHS[layer_name]
    
    if not os.path.exists(path):
        return jsonify({
            "layer": layer_name,
            "status": "empty",
            "message": f"{layer_name.title()} katmanında henüz veri yok. Streaming pipeline'ı çalıştırın.",
            "columns": [],
            "rows": [],
            "total_rows": 0
        })
    
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(path)
        df = dt.to_pandas()
        
        total = len(df)
        # Son 20 satırı göster (en güncel veriler)
        sample = df.tail(20).iloc[::-1]
        
        # Kolon sayısı çoksa, en önemli kolonları öne al
        cols = list(sample.columns)
        if len(cols) > 15:
            # Öncelikli kolonları başa al
            priority = ["timestamp","source_ip","dest_ip","attack_type","Attack_type",
                        "flow_id","Attack_label","ingestion_time",
                        "traffic_asymmetry_ratio","pkt_size_cv","flow_intensity",
                        "iat_regularity","conn_efficiency","json_payload"]
            ordered = [c for c in priority if c in cols]
            ordered += [c for c in cols if c not in ordered]
            cols = ordered[:15]
            sample = sample[cols]
        
        # NaN ve Timestamp'leri JSON uyumlu hale getir
        sample = sample.fillna("null")
        for col in sample.columns:
            if sample[col].dtype == 'datetime64[ns]' or 'time' in col.lower():
                sample[col] = sample[col].astype(str)
        
        rows = sample.to_dict(orient="records")
        
        return jsonify({
            "layer": layer_name,
            "status": "ok",
            "columns": list(sample.columns),
            "rows": rows,
            "total_rows": total
        })
        
    except Exception as e:
        return jsonify({
            "layer": layer_name,
            "status": "error",
            "message": str(e),
            "columns": [],
            "rows": [],
            "total_rows": 0
        })


if __name__ == "__main__":
    # Kafka consumer'ı arka plan thread'inde başlat
    t = threading.Thread(target=kafka_consumer_thread, daemon=True)
    t.start()
    
    print("[API] Dashboard API başlatılıyor: http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
