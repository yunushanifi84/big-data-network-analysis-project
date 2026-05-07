from kafka import KafkaConsumer
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    topic = 'iot-network-traffic'
    bootstrap_servers = ['kafka:9092']
    
    logger.info(f"Starting test consumer for topic '{topic}'")
    
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        count = 0
        logger.info("Listening for messages... Press Ctrl+C to stop.")
        
        for message in consumer:
            msg = message.value
            count += 1
            logger.info(f"[{count}] Received message:")
            logger.info(f"  - Timestamp: {msg.get('timestamp')}")
            logger.info(f"  - Flow ID: {msg.get('flow_id')}")
            logger.info(f"  - Source IP: {msg.get('source_ip')}")
            logger.info(f"  - Dest IP: {msg.get('dest_ip')}")
            logger.info(f"  - Attack Type: {msg.get('attack_type')}")
            logger.info("-" * 40)
            
    except KeyboardInterrupt:
        logger.info("Graceful shutdown requested...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info(f"Test consumer closed. Total messages received: {count}")

if __name__ == "__main__":
    main()
