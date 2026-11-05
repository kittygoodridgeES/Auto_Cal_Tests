from pathlib import Path


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