import json
import os
import time

import pika
from dotenv import load_dotenv

from api_requests import APIRequests


class AMQP_Transport:
    def __init__(self, factory_id, slot_id, transport_running, logger):
        load_dotenv()
        self.factory_id = str(factory_id)
        self.slot_id = str(slot_id)
        self.transport_running = transport_running
        self.logger = logger
        self.api = APIRequests(self.factory_id, self.slot_id)

        # "3.2.2" -> station 3, cluster 2, cutter 2;  "1.4" -> station 1, no cluster, cutter 4
        parts = self.slot_id.split(".")
        self.station = parts[0]
        if len(parts) == 3:
            self.cluster, self.cutter = parts[1], int(parts[2])
        else:
            self.cluster, self.cutter = None, int(parts[1])

    def _connect(self):
        credentials = pika.PlainCredentials(os.environ["PIKA_CREDENTIALS_USER"],
                                            os.environ["PIKA_CREDENTIALS_PASSWORD"])
        parameters = pika.ConnectionParameters(
            host=os.environ["PIKA_IP"],
            port=int(os.environ["PIKA_PORT"]),
            virtual_host=os.environ["PIKA_VIRTUAL_HOST"],
            credentials=credentials,
            heartbeat=0,
            socket_timeout=7,
        )
        return pika.BlockingConnection(parameters)

    def _queue_and_routing_key(self):
        base = f"location.uk.factory.{self.factory_id}.workstation.{self.station}"
        if self.cluster is None:
            return "auto.new.calibration", f"{base}.#"
        return "auto.new.calibration.cluster", f"{base}.cluster.{self.cluster}.#"

    def run_command(self, command, stop_event=None, timeout=300):
        """
        Send `command` ("supply", "flip", ...) to this cutter and block until the
        transport reports success for it. Raises on failure, timeout, or stop.
        """
        if not self.transport_running:
            raise RuntimeError("Transport is not running; command must be done manually")

        connection = self._connect()
        try:
            channel = connection.channel()
            queue_name, routing_key = self._queue_and_routing_key()
            channel.queue_declare(queue=queue_name, auto_delete=True)
            channel.queue_bind(queue=queue_name, exchange="transport.events",
                               routing_key=routing_key)
            self.logger.info(f"AMQP connected, listening on {routing_key}")

            # The queue is bound before the command is sent, so the result can't be missed
            self.api.command(command)
            self.logger.info(f"{command} command sent to cutter {self.cutter}")

            deadline = time.monotonic() + timeout
            for _method, _props, body in channel.consume(queue_name, auto_ack=True,
                                                         inactivity_timeout=1.0):
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError(f"{command} cancelled: program stopped")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"No {command} result for cutter {self.cutter} "
                                       f"after {timeout} s")
                if body is None:
                    continue  # nothing arrived in the last second

                data = json.loads(body)["data"]
                self.logger.info(f"Transport event: cutter {data.get('slot')}, "
                                 f"command {data.get('command')}, result {data.get('result')}")

                if data.get("command") != command or str(data.get("slot")) != str(self.cutter):
                    continue  # a different cutter or command

                result = data.get("result")
                if result == "success":
                    self.logger.info(f"{command} succeeded on cutter {self.cutter}")
                    return
                if result == 'failure':
                    raise RuntimeError(f"{command} failed on cutter {self.cutter}: {result}")
        finally:
            if connection.is_open:
                connection.close()

    def close(self):
        """Nothing to close: each run_command opens and closes its own connection."""
        pass