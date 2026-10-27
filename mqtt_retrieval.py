import paho.mqtt.subscribe as subscribe
import time

from constants import endpoints
class MQTTClient:
    def __init__(self, factory_id, slot_id):
        self.factory_id = factory_id
        self.slot_id = slot_id
        self.break_connection = False
        self.topic_prefix = self.select_cutter_topic_no(slot_id)

    def select_cutter_topic_no(self, slot_id):
        # Set cutter url on init
        if slot_id.startswith("1."):
            station = str(1)
            slot = slot_id[2:]
            cutter_url = "uk/1/workstation/1/slot/" + slot
            return cutter_url
        else:
            station = self.slot_id[:1]
            cluster = self.slot_id[2:3]
            slot = self.slot_id[4:5]
            cutter_url = "uk/"+ self.factory_id +"/workstation/"+ station +"/cluster/" + cluster +"/slot/" + slot
            return cutter_url
            
        # transport url: "uk/1/workstation/1/transport"

    def monitor(self, name: str) -> str:
        """Read the latest value of a topic, e.g. self.monitor("head_pos")."""
        if name not in endpoints.TOPICS:
            raise KeyError(f"Unknown topic '{name}'. Options: {', '.join(endpoints.TOPICS)}")
        if name in TRANSPORT_ONLY and self.topic_prefix != endpoints.TRANSPORT_PREFIX:
            raise ValueError(f"'{name}' can only be read from the transport object")

        msg = subscribe.simple(f"{self.topic_prefix}/{endpoints.TOPICS[name]}", hostname=endpoints.MQTT_HOST)
        value = msg.payload.decode("utf-8")

        if name in endpoints.LOGGED:
            label, unit = endpoints.LOGGED[name]
            print(f"Cutter {self.slot_id} {label}: {value}{unit}.")
        return value
        
 