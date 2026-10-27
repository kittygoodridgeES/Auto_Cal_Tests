import requests
import json
from dataclasses import dataclass

from constants import endpoints as endpoint
from constants import timeouts as timeouts
from constants import cutter_positioning as cutter

# ===== POST with a value ===== #
@dataclass(frozen=True)
class Setting:
    endpoint: str
    key: str | None                       # JSON key to wrap the value in; None sends the bare value
    type: type
    limits: tuple[float, float] | None = None
    allowed: frozenset | None = None


class APIRequests():
    def __init__(self, factory_id, slot_id):

        print(f"=== Initialising API connection to Cutter: {slot_id} ===")
        self.factory_id = factory_id
        self.slot_id = slot_id
        self.cutter_url = self.select_cutter_url()
    
        # Perform initial healthcheck for selected cutter
        health = self.get("healthcheck")
        if health == "OK":
            print(f"    API Health Check: {health}")
        else:
            raise Exception(f"Health Check for Cutter: {slot_id} FAILED")

    def select_cutter_url(self):
        # Set cutter url on init
        try:
            if self.slot_id.startswith("1."):
                slot = self.slot_id[2:]
                cutter_url =  "https://station1.factory"+ self.factory_id +".elasticstage.com/slot/" + slot
                return cutter_url
            else:
                slot = self.slot_id[4:5]
                cluster = self.slot_id[2:3]
                station = self.slot_id[:1]
                cutter_url = "https://cluster" + cluster + ".station" + station + ".factory" + self.factory_id + ".elasticstage.com/slot/" + slot
                return cutter_url
        
        except:
            raise TypeError(f"Cutter number must be input as string type. Input was {type(self.slot_id)}.")
 
    def _request(self, method: str, endpoint: str, data=None) -> str:
        url = self.cutter_url + endpoint
        try:
            resp = requests.request(method, url, json=data, timeout=timeouts.TIMEOUT)
        except requests.RequestException as e:
            raise ConnectionError(f"{method} {endpoint} failed: {e}") from e
 
        if not resp.ok:
            raise RuntimeError(f"{method} {endpoint} returned {resp.status_code}: {resp.text}")
        return resp.text
 
    def get(self, name: str, legend: bool = False):
        """Read a value, e.g. self.get("packml_state", legend=True) -> 4."""
        if name not in endpoints.GET_ENDPOINTS:
            raise KeyError(f"Unknown GET '{name}'. Options: {', '.join(endpoints.GET_ENDPOINTS)}")
 
        if legend and name in cutter.LEGENDS:
            title, codes = cutter.LEGENDS[name]
            print(f"==={title} Legend===")
            for code, meaning in codes.items():
                print(f"{code}: {meaning}")
 
        return self._parse(self._request("GET", endpoints.GET_ENDPOINTS[name]))
 
    def command(self, name: str) -> str:
        """Send a command with no data, e.g. self.command("stop")."""
        if name not in enspoints.COMMANDS:
            raise KeyError(f"Unknown command '{name}'. Options: {', '.join(endpoints.COMMANDS)}")
        return self._request("POST", endpoints.COMMANDS[name])
 
    def set(self, name: str, value) -> str:
        """Send a validated value, e.g. self.set("piezo_target_pos", 1.8)."""
        if name not in endpoints.SETTINGS:
            raise KeyError(f"Unknown setting '{name}'. Options: {', '.join(endpoints.SETTINGS)}")
 
        s = endpoints.SETTINGS[name]
        value = self._validate(name, s, value)
        data = {s.key: value} if s.key else value
        return self._request("POST", s.endpoint, data)
 
    def post_config(self, data: dict) -> str:
        """Send a full config dict to /config."""
        return self._request("POST", "/config", data)
 
    @staticmethod
    def _validate(name: str, s: Setting, value):
        # bool is a subclass of int in Python, so True/False would pass an int check unless excluded
        is_bool = isinstance(value, bool)
        if s.type is bool:
            ok = is_bool
        elif s.type is float:
            ok = isinstance(value, (int, float)) and not is_bool
        else:
            ok = isinstance(value, s.type) and not is_bool
        if not ok:
            raise TypeError(f"{name} must be {s.type.__name__}, not {type(value).__name__}.")
 
        if s.type is float:
            value = float(value)
 
        if s.limits:
            lo, hi = s.limits
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}. {value} is out of bounds.")
 
        if s.allowed is not None and value not in s.allowed:
            raise ValueError(f"{name} must be one of {sorted(s.allowed)}. {value} is not allowed.")
 
        return value

    
    @staticmethod
    def _parse(text: str):
        """
        '{"mode":3}'  -> 3          (single-key object: just the value)
        '{...many keys...}' -> dict (e.g. /config, returned whole)
        'OK'          -> 'OK'       (plain text that isn't JSON, e.g. healthcheck)
        """
        try:
            value = json.loads(text)
        except ValueError:
            return text.strip()
        if isinstance(value, dict) and len(value) == 1:
            return next(iter(value.values()))
        return value