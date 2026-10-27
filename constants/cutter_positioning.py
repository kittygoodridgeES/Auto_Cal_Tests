FIRST_CALIBRATION_HEADWEIGHT = (4.5, 6.5)   # Range in which headweight should sit for the first calibration on a cutter (no prior performance information)
DROP_IN_AIM = 90                            # um which a drop in should aim for to protect the diamond - safe > 70
SHIM_OFFSET = 0.5                           # mm offset to ensure shim captures desired headweight value
ROUND_SHIM = 2                              # number of decimal places to round the shim value for the loadcell to
MANUAL_DISK_CYCLE = {"No Disk": "A_side", "A_side": "B_side", "B_side": "No Disk"}
DEFAULT_GW = 50                             # default groove width set on the cutter (defualt headweight is measured at this value)
ABS_ERROR_MARGIN = 5                        # max allowed |measured - target| (um)


PIEZO_GROOVES = 3                            # grooves at the start of each band affected by the piezo move
NUMBER_OF_VAL_SAMPLES = 4                    # stationary photos taken after the cut
VALIDATION_S = 10 * REV_S                    # cutting time at each target width (s)
VALIDATION_WIDTHS = [65, 45, 55, 65, 75, 95, 105]   # 77 first for a safe drop-in without lead-in, disk will most likely swarf consistently cutting at 125 so set max at 110
DROPIN_PITCH = 75.0                          # pitch where cutter can drop within the camera view
REV_S = 1.8                                  # one platter revolution = one groove (s)

DROP_IN_AIM = 90.0                    # um width that the cutter will aim to drop in at
DELAY_CUT_TO_FRAME = 0.368            # time delay from the cuttin head to being visible in the camera frame
MEASURE_S = 5 * 1.8                   # recording time at each position (s)
SETTLE_S = 0.2                        # ignore frames for this long after a move finishes (s)

PIEZO_UPPER_LIMIT = 3.2                # maximum mm position of the piezo
PIEZO_LOWER_LIMIT = 0.5                # minimum mm position of the piezo
MAX_SAFE_GW = 130                       # maximum groove width requested on cutter
MIN_SAFE_GW = 40                        # minimum groove width requested on cutter
HEAVY_STEP = 0.2                      # after a heavy-limit hit, go this far below the drop-in position
NO_GROOVE_STEP = 0.5                  # if no grooves are seen for a set window (timeouts) increase the piezo by this much (capped at PIEZO_UPPER_LIMIT), once


MIN_SAMPLES = 20                      # 5-frame averages needed at a position before moving on
MERGE_RATIO = 1.7                     # width jumping to >= this many times the last good width = merged grooves
MERGE_WINDOW_S = 0.5                  # ...if the jump happens within this many seconds

HEAD_SETTLE_TIME = 20                   # time taken for headweight to settle on the loadcell
HEAD_DROP_TIME = 5                      # rough time taken for head to drop once command is sent

 
# ===== PACKML codes ===== #
PIEZO_CMD_IDS = {
    0: "None", 
    1: "Home", 
    6: "Stop", 
    7: "Jog Positive",
    8: "Jog Negative", 
    14: "Move Position", 
    15: "Move Relative",
}

LEGENDS = {
    "piezo_cmd_id": ("Piezo Cmd ID", PIEZO_CMD_IDS),
    "packml_state": ("PackML State", PACKML_STATES),
}

PACKML_MODES = {
    0: "ModeNone",
    1: "ModeProduction",
    2: "ModeManual",
    3: "ModeMaintenance",
    4: "ModeFormatChange",
    5: "ModeCleaning",
    6: "ModeSetUp",
    7: "ModeEmptyOut",
    8: "ModeDryRun",
    9: "ModeStandBy",
    }
 
PACKML_COMMANDS = {
  0: "CmdNone",
  1: "CmdReset",
  2: "CmdStart",
  3: "CmdStop",
  4: "CmdHold",
  5: "CmdUnhold",
  6: "CmdSuspend",
  7: "CmdUnsuspend",
  8: "CmdAbort",
  9: "CmdClear",
  10: "CmdComplete" ,
}

PACKML_STATES =  {
  "PML_StateNone" = 0,
  "PML_StateClearing" = 1,
  "PML_StateStopped" = 2,
  "PML_StateStarting" = 3,
  "PML_StateIdle" = 4,
  "PML_StateSuspended" = 5,
  "PML_StateExecute" = 6,
  "PML_StateStopping" = 7,
  "PML_StateAborting" = 8,
  "PML_StateAborted" = 9,
  "PML_StateHolding" = 10,
  "PML_StateHeld" = 11,
  "PML_StateUnholding" = 12,
  "PML_StateSuspending" = 13,
  "PML_StateUnsuspending" = 14,
  "PML_StateResetting" = 15,
  "PML_StateCompleting" = 16,
  "PML_StateCompleted" = 17,
}