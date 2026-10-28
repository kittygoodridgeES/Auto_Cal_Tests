

# ------- API endpoints ------------------------
GET_ENDPOINTS = {
    "healthcheck":       "/healthcheck",
    "pitches":           "/cut/pitches",
    "widths":            "/cut/widths",
    "head_drop_pos":     "/config/head_drop_position",
    "piezo_target_pos":  "/config/piezo_par_position",
    "piezo_cmd_id":      "/config/piezo_cmd_id",
    "default_piezo_pos": "/config/default_piezo_position",
    "packml_state":      "/state/packml",
    "packml_mode":       "/state/mode",
    "config":             "/config",
}

COMMANDS = {
    "config_apply":     "/config/apply",
    "piezo_reset":      "/piezo/reset",
    "mode_production":  "/state/mode/production",
    "mode_manual":      "/state/mode/manual",
    "mode_maintenance": "/state/mode/maintenance",
    "mode_setup":       "/state/mode/setup",
    "start":            "/state/command/start",
    "reset":            "/state/command/reset",
    "stop":             "/state/command/stop",
    "flip":             "/flip",
    "supply":           "/supply",
}

PIEZO_CMD_IDS = {
  0: "None",
  1: "Home",
  6: "Stop",
  7: "Jog Positive",
  8: "Jog Negative",
  14: "Move Position",
  15: "Move Relative",
}



#----------- MQTT topics ----------------------
MQTT_HOST = "mqtt.factory1.elasticstage.com"
TRANSPORT_PREFIX = "uk/1/workstation/1/transport"

TOPICS = {
    "piezo_actual":             "piezo_actual",
    "piezo_target":             "piezo_tar",
    "piezo_cmd":                "piezo_command",
    "piezo_state":              "piezo_motor_state",
    "piezo_default_pos":        "default_piezo_position",
    "piezo_cmd_progress":       "piezo_command_progress",
    "piezo_cmd_active_id":      "piezo_command_active_id",
    "piezo_cmd_active_tar_pos": "piezo_command_active_par_position",
    "head_pos":                 "head_position",
    "packml_execute":           "packml_execute",
    "packml_state":             "packml_state",
    "packml_mode":              "packml_mode",
    "vacuum_cmd":               "vacuum_command",
    "platter_spd":              "platter_speed",
    "head_weight":              "weight",
    "record_present":           "record_present",
    "transport_target_cutter":  "destination_cutter",
}
LOGGED = {
    "piezo_target": ("Piezo Target", "mm"),
    "piezo_cmd":    ("Piezo Cmd", ""),
    "piezo_state":  ("Piezo State", ""),
    "head_pos":     ("Head Pos", ""),
}

TRANSPORT_ONLY = {"transport_target_cutter"}