import pandas as pd
import json
import time
import argparse
from kafka import KafkaProducer
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_producer(bootstrap_servers):
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Connected to Kafka broker at {bootstrap_servers}")
        return producer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="IoT Network Traffic Kafka Producer")
    parser.add_argument("--rate", type=int, default=50, help="Messages per second")
    parser.add_argument("--broker", type=str, default="kafka:9092", help="Kafka broker address")
    parser.add_argument("--topic", type=str, default="iot-network-traffic", help="Kafka topic name")
    parser.add_argument("--file", type=str, default="/app/data/raw/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv", help="Path to CSV file")
    
    args = parser.parse_args()
    
    producer = create_producer(args.broker)
    if not producer:
        return

    logger.info(f"Starting to produce messages to topic '{args.topic}' at {args.rate} msg/sec")
    
    try:
        chunksize = 10000
        count = 0
        
        for chunk in pd.read_csv(args.file, chunksize=chunksize, low_memory=False):
            for index, row in chunk.iterrows():
                message = row.dropna().to_dict()
                
                # Enforce required fields
                # 1. timestamp
                message['timestamp'] = datetime.utcnow().isoformat() + "Z"
                
                # 2. source_ip / dest_ip
                if 'source_ip' not in message:
                    message['source_ip'] = message.get('ip.src_host', message.get('arp.src.proto_ipv4', '0.0.0.0'))
                if 'dest_ip' not in message:
                    message['dest_ip'] = message.get('ip.dst_host', message.get('arp.dst.proto_ipv4', '0.0.0.0'))
                    
                # 3. attack_type
                if 'attack_type' not in message:
                    message['attack_type'] = message.get('Attack_type', 'Normal')
                
                # 4. flow_id
                if 'flow_id' not in message:
                    # Create a simple flow_id using src and dst
                    src_port = message.get('tcp.srcport', message.get('udp.port', '0'))
                    dst_port = message.get('tcp.dstport', '0')
                    message['flow_id'] = f"{message['source_ip']}:{src_port}-{message['dest_ip']}:{dst_port}"
                
                # Send to Kafka
                key = str(message.get('source_ip', str(count))).encode('utf-8')
                producer.send(args.topic, key=key, value=message)
                
                count += 1
                if count % 1000 == 0:
                    logger.info(f"Sent {count} messages...")
                    
                if args.rate > 0:
                    time.sleep(1.0 / args.rate)
                    
    except KeyboardInterrupt:
        logger.info("Graceful shutdown requested...")
    except FileNotFoundError:
         logger.error(f"CSV file not found at {args.file}. Please check the path.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if producer:
            producer.flush()
            producer.close()
        logger.info(f"Finished. Total messages sent: {count}")

if __name__ == "__main__":
    main()
