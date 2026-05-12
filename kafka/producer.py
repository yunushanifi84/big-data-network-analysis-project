import csv
import json
import time
import argparse
import math
from kafka import KafkaProducer
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_producer(bootstrap_servers, max_retries=30, retry_interval=5):
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                buffer_memory=16777216,
                batch_size=16384,
                linger_ms=10,
            )
            logger.info(f"Connected to Kafka broker at {bootstrap_servers}")
            return producer
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_interval)
    logger.error(f"Could not connect to Kafka after {max_retries} attempts")
    return None


def build_message(row):
    """CSV satırından Kafka mesajı oluşturur. Boş değerleri atlar."""
    message = {k: v for k, v in row.items() if v is not None and v != ''}

    for key, val in message.items():
        try:
            if '.' in val:
                message[key] = float(val)
            else:
                message[key] = int(val)
        except (ValueError, TypeError):
            pass

    message['timestamp'] = datetime.utcnow().isoformat() + "Z"

    if 'source_ip' not in message:
        message['source_ip'] = message.get('ip.src_host', message.get('arp.src.proto_ipv4', '0.0.0.0'))
    if 'dest_ip' not in message:
        message['dest_ip'] = message.get('ip.dst_host', message.get('arp.dst.proto_ipv4', '0.0.0.0'))
    if 'attack_type' not in message:
        message['attack_type'] = message.get('Attack_type', 'Normal')
    if 'flow_id' not in message:
        src_port = message.get('tcp.srcport', message.get('udp.port', '0'))
        dst_port = message.get('tcp.dstport', '0')
        message['flow_id'] = (
            f"{message['source_ip']}:{src_port}"
            f"-{message['dest_ip']}:{dst_port}"
            f"-{message['timestamp']}"
        )

    return message


def main():
    parser = argparse.ArgumentParser(description="IoT Network Traffic Kafka Producer")
    parser.add_argument("--rate", type=int, default=50, help="Messages per second")
    parser.add_argument("--broker", type=str, default="kafka:9092", help="Kafka broker address")
    parser.add_argument("--topic", type=str, default="iot-network-traffic", help="Kafka topic name")
    parser.add_argument("--file", type=str, default="/app/data/raw/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv", help="Path to CSV file")
    parser.add_argument("--max-messages", type=int, default=0, help="Maximum number of messages to send (0 = unlimited)")
    parser.add_argument("--no-delay", action="store_true", help="Disable rate limiting (send as fast as possible)")

    args = parser.parse_args()

    producer = create_producer(args.broker)
    if not producer:
        return

    max_messages = args.max_messages
    logger.info(f"Topic: '{args.topic}' | Rate: {args.rate} msg/sec | "
                f"Max: {max_messages if max_messages > 0 else 'unlimited'}")

    count = 0
    try:
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                message = build_message(row)
                key = str(message.get('source_ip', str(count))).encode('utf-8')
                producer.send(args.topic, key=key, value=message)

                count += 1
                if count % 5000 == 0:
                    producer.flush()
                    logger.info(f"Sent {count} messages...")

                if max_messages > 0 and count >= max_messages:
                    logger.info(f"Reached max message limit ({max_messages}).")
                    break

                if not args.no_delay and args.rate > 0:
                    time.sleep(1.0 / args.rate)

    except KeyboardInterrupt:
        logger.info("Graceful shutdown requested...")
    except FileNotFoundError:
        logger.error(f"CSV file not found at {args.file}.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if producer:
            producer.flush()
            producer.close()
        logger.info(f"Finished. Total messages sent: {count}")


if __name__ == "__main__":
    main()
