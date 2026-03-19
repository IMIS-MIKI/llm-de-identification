import os
import json
import sys

from datetime import datetime
from dotenv import load_dotenv
from kafka.consumer import KafkaConsumer
from kafka.producer import KafkaProducer

from runner.src.logging_config import logger
from runner.src.util import process_text, convert_to_viewer_format

load_dotenv("runner/.env", verbose=True)
load_dotenv(verbose=True)


def get_modus_props():
    bootstrap_server = os.environ['BOOTSTRAP_SERVER'].split(',')
    max_poll = int(os.environ['MAX_POLL_RECORDS'])
    in_topic = os.environ['INCOMING_INDEX_TOPIC_NAME']
    out_topic = os.environ['OUTGOING_INDEX_TOPIC_NAME']
    consumer_group = os.environ['INDEX_CONSUMER_GROUP']
    return bootstrap_server, max_poll, in_topic, out_topic, consumer_group


def send_error(message, error):
    bootstrap_server, _, _, _, _ = get_modus_props()
    producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                             bootstrap_servers=bootstrap_server)

    res = dict()
    res['dateCreated'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    res['message'] = message.value
    res['error'] = str(error)

    producer.send(topic=os.environ['ERROR_TOPIC_NAME'], value=res)
    producer.flush()


def start_consumer():
    bootstrap_servers, max_poll, in_topic, out_topic, consumer_group = get_modus_props()
    print(f"=== Starting LLM Anonymization service ===")
    print(f"Bootstrap servers: {bootstrap_servers}")
    print(f"Subscribing to topic: {in_topic}")
    print(f"Consumer group: {consumer_group}")
    sys.stdout.flush()  # Wichtig!

    try:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=consumer_group,
            max_poll_records=max_poll,
            value_deserializer=lambda x: json.loads(x.decode("utf-8"))
        )
        print(f"Kafka consumer created successfully")
        sys.stdout.flush()

    except Exception as error:
        print('ERROR ' + str(error.__module__) + ': ' + str(error), file=sys.stderr)
        exit()

    consumer.subscribe([in_topic])
    print(f"Subscribed to topic: {in_topic}")
    print(f"Waiting for messages...")
    sys.stdout.flush()
    for message in consumer:
        print(f"Received message from partition {message.partition}, offset {message.offset}")
        sys.stdout.flush()
        try:
            consumer.commit()
            payload = message.value

            logger.debug('Payload: ' + str(payload))

            metadata = payload['metadata']
            text = payload['text']

            # ====== Here comes the processing ======
            pseudo_text, runtime = process_text(text)

            res = convert_to_viewer_format(text, pseudo_text, runtime)

            res['metadata'] = metadata

            logger.debug('Response: ' + str(res))

            # https://forum.confluent.io/t/what-should-i-use-as-the-key-for-my-kafka-message/312
            producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                     bootstrap_servers=bootstrap_servers)
            producer.send(topic=out_topic, value=res)
            producer.flush()

        except Exception as error:
            print('ERROR ' + str(error), file=sys.stderr)
            send_error(message, error)


if __name__ == "__main__":
    start_consumer()
