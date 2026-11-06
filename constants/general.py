from pathlib import Path
import re


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"

#analysis constants
N_SD = 3               # error bars = ± this many standard deviations
MIN_N = 5              # ignore grooves with fewer rows than this
FIT = True             # fit piezo = gradient * width + intercept and print it
EXCLUDE_GROOVES = []   # groove numbers to leave out of the plot and fit, e.g. [2, 3]
SAMPLE_NO = 4          # number of stationary samples taken from validation side


DEFAULT_DROP_IN_PIEZO_POS = 1.85
REJECT_OUTLIERS = True
OUTLIER_K = 3.5
OUTLIER_MIN_RESID = 0.05
MIN_FIT_GROOVES = 3

# e.g. groove_piezo_20261102_162352_plot.png
GROOVE_PLOT_PATTERN = re.compile(r"groove_piezo_(\d{8}_\d{6})_plot\.png")

# ===== Database ===== #
# Change this if the table has a different name
CALIBRATION_TABLE = "calibration"

# end_time is filled in by the database, so it isn't sent
CALIBRATION_INSERT_SQL = f"""
    INSERT INTO {CALIBRATION_TABLE}
        (laptop, slot, transport_on, diamond_id, mmum, y_int, "pass", discs_used, default_headweight, time_taken, correlation_graph, validation_graph, maintenance_narrative)
    VALUES
        (%(laptop)s, %(slot)s, %(transport_on)s, %(diamond_id)s, %(mmum)s, %(y_int)s, %(pass)s,
         %(discs_used)s, %(default_headweight)s, %(time_taken)s, %(correlation_graph)s, %(validation_graph)s, %(maintenance_narrative)s)
"""