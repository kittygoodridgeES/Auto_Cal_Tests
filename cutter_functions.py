import time
from api_requests import APIRequests
from mqtt_retrieval import MQTTClient

from constants import cutter_positioning as cutter
from constants import timeouts


def move_piezo(api, mqtt, targ, timeout=timeouts.PIEZO_MOVE_TIMEOUT):
    """Move the piezo to targ. Raises TimeoutError if it doesn't get there in time,
    and always tries to leave piezo_cmd_id at 0 on failure."""
    targ = round(float(targ), 2)

    def at_target():
        actual = float(mqtt.monitor("piezo_actual"))
        print(f"Waiting for position {targ}, actual {actual!r}")
        return abs(actual - targ) <= 0.02

    try:
        # prepare piezo for movement
        print("set piezo_cmd_id 14 ->", api.set("piezo_cmd_id", 14))
        wait_for(lambda: api.get("piezo_cmd_id") == 14, "piezo cmd id 14", api,
                 timeout=timeouts.PIEZO_CMD_TIMEOUT, poll=0.5)

        # move piezo
        print(f"set piezo_target_pos {targ} ->", api.set("piezo_target_pos", targ))
        wait_for(at_target, f"piezo to reach {targ}", api, timeout=timeout, poll=0.5)

        # stop piezo
        print("set piezo_cmd_id 0 ->", api.set("piezo_cmd_id", 0))
        wait_for(lambda: api.get("piezo_cmd_id") == 0, "piezo cmd id 0", api,
                 timeout=timeouts.PIEZO_CMD_TIMEOUT, poll=0.5)

    except Exception:
        # Don't leave the piezo in move mode
        try:
            api.set("piezo_cmd_id", 0)
        except Exception:
            pass
        raise

def wait_for(condition, description, api,  timeout=timeouts.STATE_TIMEOUT, poll=0.1):
    """
    Poll `condition()` until it returns True, or raise TimeoutError
    if it hasn't happened within `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for: {description} "
                f"(packml_state={api.get('packml_state')}, "
                f"packml_mode={api.get('packml_mode')})"
            )
        time.sleep(poll)


def find_default_headweight(api, mqtt, default_piezo_pos):
    """
    Calculates the default headweight initially at the default piezo position of 50um
    If there is not enough weight on the loadcell at thAis point lower the head further and calculate the 
    new default piezo position, default weight and default groove width
    returns True/False dependant on whether the default headweight at 50um can be used, The default headweight
    """
    #put into maintenance mode
    if api.get("packml_mode") != 3:
        api.command("mode_maintenance")
        wait_for(lambda: api.get("packml_mode") == 3, "cutter to maintenance mode", api)

    #stop cutter
    api.command("stop")
    wait_for(lambda: api.get("packml_state") == 2, "cutter to stop", api)

    #reset cutter
    api.command("reset")
    wait_for(lambda: api.get("packml_state") == 4, "reset to finish (IDLE)", api)
    
    #drop head
    api.set("head_ctrl", 1)
    time.sleep(cutter.HEAD_DROP_TIME)

    #move piezo to default piezo pos
    move_piezo(api, mqtt, default_piezo_pos)

    #check if headweight >3g

    #measure headweight
    time.sleep(cutter.HEAD_SETTLE_TIME)
    headweight = mqtt.monitor("head_weight")

    if float(headweight) > 3:
        #headweight greater than 3g return to be submitted to config
        return_value = [True, headweight, None]
    else:
        pos = default_piezo_pos + 0.2
        #head is not far enough on weight continue iteratively downward until head is greater than 3g
        while headweight < 3:
            move_piezo(api, mqtt, pos)
            time.sleep(cutter.HEAD_SETTLE_TIME)
            headweight = mqtt.monitor("head_weight")
            pos += 0.2
        piezo_position = mqtt.monitor("piezo_actual")
        return_value = [False, headweight, piezo_position]


    #lift head
    api.set("head_ctrl", 0)
    time.sleep(cutter.HEAD_DROP_TIME)

    #reset cutter
    api.command("reset")
    wait_for(lambda: api.get("packml_state") == 4, "reset to finish (IDLE)", api)
    return return_value

# api = APIRequests("1", "3.2.4")
# mqtt= MQTTClient("1", "3.2.4")
# print(find_default_headweight(api, mqtt, 1.9806148174731648))
# # move_piezo(api, mqtt, 2.65)