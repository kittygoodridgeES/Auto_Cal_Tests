import logging
import logging.handlers
import queue
import threading
import time
import webbrowser
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from airtable_retrieval import AirtableReader
from mqtt_retrieval import MQTTClient
from transport_commands import AMQP_Transport
from api_requests import APIRequests
from sideA import CalibrationSide
from linear_analysis_sideA import LinearAnalysis
from cutter_functions import find_default_headweight
from validation import ValidationSide

import cv2
import numpy as np
from PIL import Image, ImageTk
import yaml
import os
import dotenv


# ===== import constants  ===== #
from constants import gui_presentation as gui
from constants import cutter_positioning as cutter
from constants import timeouts as timeout
from constants import general as gen


#disk position tracking: "No Disk", "A_side" or "B_side"
disk_position = "No Disk"

# Manual mode (transport not operational): each manual task moves the disc one step on
#   No Disk --place disc A up--> A_side --flip--> B_side --remove--> No Disk
length_change = None

#load .env file
dotenv.load_dotenv()

#import yaml file for config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


#initialise universal classes
airtable = AirtableReader(
    tables={
        "Jobs": os.environ["JOB_TABLE_ID"],
        "Location Update Events": os.environ["AT_DIAMOND_BASE_ID"],
        "Length Update Events": os.environ["AT_DIAMOND_BASE_ID"],
    },
)


def factory_name(factory: str) -> str:
    entry = config["FACTORIES"][factory]
    return entry.get("name", str(factory)) if isinstance(entry, dict) else str(entry)


def stations_for(factory: str) -> dict:
    entry = config["FACTORIES"][factory]
    if isinstance(entry, dict) and "stations" in entry:
        return entry["stations"]
    return config["STATION_LAYOUT"]


def station_layout(factory: str, station: str) -> dict:
    return stations_for(factory)[station]


def cutter_count(layout: dict) -> int:
    """Cutters on the station, or per cluster for stations with clusters."""
    return layout.get("cutters", layout.get("slots"))


# ===== Logging ===== #
def setup_logging(name="wizard") -> logging.Logger:
    """Console logging from startup. Anything logged before the log file exists
    is held in memory and written to the file once start_file_logging() runs."""
    logger = logging.getLogger(name)
    if logger.handlers:            # already set up (e.g. called twice)
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False       # avoid duplicate lines via the root logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(gen.LOG_FORMAT))
    logger.addHandler(console_handler)

    # Buffers early records; never flushes on its own because no target is set yet
    buffer = logging.handlers.MemoryHandler(capacity=10000, flushLevel=logging.CRITICAL + 1)
    buffer.set_name("startup_buffer")
    logger.addHandler(buffer)
    return logger


def start_file_logging(logger: logging.Logger, file_stem: str, log_dir: Path | None = None):
    """Create logs/<stem>_<timestamp>/<stem>_<timestamp>.log and write the buffered
    startup lines into it. Returns (log_path, log_folder)."""
    log_dir = log_dir or gen.LOG_DIR
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            path = Path(h.baseFilename)
            return path, path.parent       # already logging to a file

    stamp = f"{file_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_folder = log_dir / stamp
    log_folder.mkdir(parents=True, exist_ok=True)   # creates logs\ and the run folder
    log_path = log_folder / f"{stamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(gen.LOG_FORMAT))

    # Write out anything logged before this point, then drop the buffer
    for h in list(logger.handlers):
        if isinstance(h, logging.handlers.MemoryHandler):
            h.setTarget(file_handler)
            h.flush()
            logger.removeHandler(h)
            h.close()

    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")
    return log_path, log_folder

# ===== Shared state ===== #
@dataclass
class Selection:
    factory: str | None = None
    station: str | None = None
    cluster: str | None = None
    cutter: str | None = None
    transport_ok: bool | None = None

    def layout(self) -> dict | None:
        if self.factory is None or self.station is None:
            return None
        return station_layout(self.factory, self.station)

    def has_clusters(self) -> bool:
        layout = self.layout()
        return layout is not None and "clusters" in layout

    def is_complete(self) -> bool:
        if (self.factory is None or self.station is None or self.cutter is None
                or self.transport_ok is None):
            return False
        return self.cluster is not None or not self.has_clusters()

    @property
    def machine_id(self) -> str | None:
        """station.cluster.cutter, or station.cutter for stations without clusters.
        None until the location has been fully chosen on the selection page."""
        if self.station is None or self.cutter is None:
            return None
        if self.cluster is not None:
            return f"{self.station}.{self.cluster}.{self.cutter}"
        return f"{self.station}.{self.cutter}"

    def summary(self) -> str:
        parts = []
        if self.factory:
            parts.append(f"Factory {factory_name(self.factory)}")
        if self.station:
            parts.append(f"Station {self.station}")
        if self.cluster:
            parts.append(f"Cluster {self.cluster}")
        if self.cutter:
            parts.append(f"Cutter {self.cutter}")
        if self.transport_ok is not None:
            parts.append(f"Transport {'operational' if self.transport_ok else 'not operational'}")
        return ", ".join(parts) if parts else "Nothing selected yet"


class FrameBuffer:
    """Holds the latest camera frame. Written by a worker thread, read by the GUI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame

    def clear(self):
        self.set(None)


@dataclass
class WorkerContext:
    """Everything a background task gets handed."""
    selection: Selection
    answers: dict                 # Yes/No answers from confirm pages, keyed by page key
    values: dict                  # values read on pages, e.g. values["diamond_id"]
    stop_event: threading.Event
    logger: logging.Logger
    frames: FrameBuffer
    get_mqtt: Callable[[], "MQTTClient"]
    get_transport: Callable[[], "AMQP_Transport"]
    get_api: Callable[[], "APIRequests"]
    # Set by TaskPage: ask_operator(message) shows the message with a CONFIRM button
    # and blocks until the operator presses it (raises if STOP PROGRAM is pressed)
    ask_operator: Callable[[str], None] | None = None


    def log(self, msg: str):
        self.logger.info(msg)


class QueueLogHandler(logging.Handler):
    """Sends log records into a queue so the GUI can display them."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.q.put(self.format(record))


# ===== App ===== #
class WizardApp(tk.Tk):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.selection = Selection()
        self.answers: dict[str, bool] = {}
        self.values: dict = {}
        self.current_page = None
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._ui_queue: queue.Queue = queue.Queue()
        self.assets_dir = Path(__file__).resolve().parent / "assets"
        self._mqtt = None
        self._mqtt_lock = threading.Lock()
        self._transport = None
        self._transport_lock = threading.Lock()
        self._api = None
        self._api_lock = threading.Lock()


        self.title("Setup Wizard")
        self.configure(background=gui.COLOR_BG)
        win_w = min(1400, self.winfo_screenwidth() - gui.HEIGHT_MARGIN_REMOVED)
        win_h = min(850, self.winfo_screenheight() - gui.HEIGHT_MARGIN_REMOVED)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self.stop_program)

        self._configure_style()
        self.logo_photo = self._load_logo()

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        self._build_pages()
        self.show_page("instructions")
        self._poll_ui_queue()

    def get_api(self) -> APIRequests:
        """Create the API client for the selected machine on first use, then reuse it."""
        with self._api_lock:
            if self._api is None:
                sel = self.selection
                self.logger.info(f"Creating API client for factory {sel.factory}, slot {sel.machine_id}")
                self._api = APIRequests(str(sel.factory), sel.machine_id)
            return self._api

    def get_transport(self) -> AMQP_Transport:
        """Create the AMQP transport connection for the selected machine on first use, then reuse it."""
        with self._transport_lock:
            if self._transport is None:
                sel = self.selection
                self.logger.info(f"Connecting AMQP transport for factory {sel.factory}, slot {sel.machine_id}")
                self._transport = AMQP_Transport(
                    factory_id=sel.factory,
                    slot_id=sel.machine_id,
                    transport_running=sel.transport_ok,
                    logger=self.logger,
                )
            return self._transport
        
    def _close_transport(self):
        if self._transport is None:
            return
        try:
            self._transport.close()   # or whatever AMQP_Transport's shutdown method is
        except Exception:
            self.logger.exception("Error closing AMQP transport")
        self._transport = None

    def get_mqtt(self) -> MQTTClient:
        """Create the MQTT client for the selected machine on first use, then reuse it."""
        with self._mqtt_lock:
            if self._mqtt is None:
                sel = self.selection
                self.logger.info(f"Connecting MQTT for factory {sel.factory}, machine {sel.machine_id}")
                self._mqtt = MQTTClient(sel.factory, sel.machine_id)
            return self._mqtt

    def _close_mqtt(self):
        if self._mqtt is None:
            return
        try:
            self._mqtt.disconnect()   # or whatever your MQTTClient's shutdown method is
        except Exception:
            self.logger.exception("Error closing MQTT client")
        self._mqtt = None

    def report_callback_exception(self, exc, val, tb):
        """Tkinter normally just prints errors from button clicks and after() callbacks
        to the console. Send them to the log file instead."""
        self.logger.error("Unhandled error in GUI callback", exc_info=(exc, val, tb))

    # --- threading helpers --- #
    def run_in_background(self, target, *args) -> threading.Thread:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        self.threads.append(t)
        return t

    def call_in_ui(self, fn, *args):
        """Tkinter isn't thread-safe: worker threads use this to ask the GUI thread to run fn."""
        self._ui_queue.put((fn, args))

    def _poll_ui_queue(self):
        try:
            while True:
                fn, args = self._ui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.after(50, self._poll_ui_queue)

    def make_context(self, frames: FrameBuffer | None = None) -> WorkerContext:
        return WorkerContext(self.selection, self.answers, self.values, self.stop_event,
                             self.logger, frames or FrameBuffer(), self.get_mqtt,
                             self.get_transport, self.get_api)

    # --- setup --- #
    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=gui.COLOR_BG)
        style.configure("TLabel", background=gui.COLOR_BG, foreground=gui.COLOR_TEXT, font=gui.FONT_BODY)
        style.configure("Header.TFrame", background=gui.COLOR_HEADER_BG)
        style.configure("HeaderTitle.TLabel", background=gui.COLOR_HEADER_BG,
                        foreground=gui.COLOR_HEADER_TEXT, font=gui.FONT_HEADER_TITLE)
        style.configure("TLabelframe", background=gui.COLOR_CARD_BG, relief="flat", borderwidth=1)
        style.configure("TLabelframe.Label", background=gui.COLOR_CARD_BG,
                        foreground=gui.COLOR_TEXT, font=gui.FONT_BODY_BOLD)
        style.configure("Card.TLabel", background=gui.COLOR_CARD_BG)

    def _load_logo(self):
        try:
            img = Image.open(self.assets_dir / "eslogo.png")
            img.thumbnail((170, 60))
            return ImageTk.PhotoImage(img)
        except Exception:
            self.logger.warning("Could not load eslogo.png")
            return None

    def _build_pages(self):
        assets = self.assets_dir

        #Page 0: Setup Instructions
        self.pages["instructions"] = InfoPage(
            self.container, self,
            title="Setup",
            body="Place camera on the mount as shown. \nMake sure it is at the bottom of the rail and press lever downwards. \nConnect USB to Calibration Laptop and plug in camera light. \nMake sure camera cable is not blocking cutter level sensor.  \nPress Next to continue.",
            image_path = [assets/"setup.png"],               # e.g. [assets / "step1.png"]
            next_target="selection",
        )

        # Page 1: Data Entry
        self.pages["selection"] = SelectionPage(self.container, self, next_target="check_state")

        # Page 1.5: Check the cutter state
        self.pages["check_state"] = CheckPage(
            self.container, self,
            title="Cutter State",
            body="The cutter must be out of production and in an idle state to continue.",
            check=read_cutter_state,
            next_target="confirm",
            pass_message="Cutter is in idle. Press Next to continue.",
            fail_message="Cutter is not in correct state - must be out of production and in idle.\n\n"
                         "Change the cutter state, then press Reload.",
        )

        # Page 2: diamond ID is read when the page opens and on Reload; operator confirms it
        self.pages["confirm"] = ReloadConfirmPage(
            self.container, self,
            title="Diamond ID",
            body="Check the diamond ID below matches the diamond currently in the cutter.",
            value_label="Diamond ID",
            read_value=read_diamond_id,
            value_key="diamond_id",
            question="Is this the diamond currently in the cutter?",
            key="ready",
            next_target="live_value",          # only used if decide_next is None
            no_message="Fill out the diamond location form to correct it, then press Reload.",
            form_url=config.get("DIAMOND_FORM_URL"),
            decide_next=choose_after_diamond,
        )

        # Page 2.5: Set headweight
        lo, hi = cutter.FIRST_CALIBRATION_HEADWEIGHT
        self.pages["live_value"] = ReloadValuePage(
            self.container, self,
            title="Headweight",
            body=f"Adjust until the value is between {lo} and {hi} g, then press Reload.",
            value_label="Head weight",
            unit=" g",
            decimals=2,
            read_value=read_live_value,
            valid_range=cutter.FIRST_CALIBRATION_HEADWEIGHT,
            next_target="supply",
        )

        # Page 3: supply page
        self.pages["supply"] = TaskPage(
            self.container, self,
            title="Supply",
            body="Supplying cutter with disc...",
            task=supply_task,
            next_target="Calibration",
            manual_instructions={
                "No Disk": "Please place a disc in the cutter with side A facing up.\n\nPress Next when done.",
                "A_side": "Please manually flip the disk in the cutter \n\nPress Next when done.",
                "B_side": "Please manually remove the disk currently in the cutter"
            },
            manual_next_position=cutter.MANUAL_DISK_CYCLE,
        )

        # Page 4: calibration
        self.pages["Calibration"] = CameraLogPage(
            self.container, self,
            title="Calibration",
            worker=run1_worker,
            next_target=choose_after_calibration,
        )

        # Page 4.5: manual adjustment (only if the analysis asks for one)
        self.pages["manual_adjust"] = InfoPage(
            self.container, self,
            title="Manual Adjustment",
            body=manual_adjust_text,
            next_target="transition",
        )

        # Page 5: Calculate default headweight
        self.pages["dataCalcs"] = ImageTaskPage(
            self.container, self,
            title="Calibration Analysis",
            body="Calculating mm/um value and default headweight for cutter config...",
            task=measure_and_set_headweight,
            next_target="transition",
            image_path=get_image,  
        )

        #Page 5.5 Shim loadcell
        self.pages["LoadCellShim"] = RepeatCheckPage(
            self.container, self,
            title="Load Cell Shim",
            body=load_cell_shim_text,
            action=measure_and_set_headweight,
            next_target="transition",
            button_text="OK",
            pass_message="Default headweight set and config applied. Press Next to continue.",
        )

        # Page 6: switch over. Gets the disc ready for the next run based on disk_position,
        # then goes to the destination chosen after the calibration:
        # dataCalcs -> validation, manual_adjust -> Calibration
        self.pages["transition"] = TaskPage(
            self.container, self,
            title="Switch Over",
            body="Preparing the disc for the next run...",
            task=transition_task,
            next_target=choose_after_transition,
            manual_instructions={
                "No Disk": "No disc in the cutter. Please place a disc in the cutter with side A facing up."
                           "\n\nPress Next when done.",
                "A_side": "Please manually flip the disc in the cutter so side B is facing up."
                          "\n\nPress Next when done.",
                "B_side": "Please manually remove and reject the disc currently in the cutter, "
                          "then place a new disc with side A facing up.\n\nPress Next when done.",
            },
            # Same cycle, except B_side is remove (-> No Disk) then place a new disc (-> A_side)
            manual_next_position={**cutter.MANUAL_DISK_CYCLE, "B_side": "A_side"},
        )

        # Page 7: validation (moves on automatically when finished)
        self.pages["validation"] = CameraLogPage(
            self.container, self,
            title="Validation",
            worker=validation_worker,
            next_target="complete",
        )

        # Page 8: Manual reject
        self.pages["complete"] = InfoPage(
            self.container, self,
            title="Remove Disk",
            body="Please Remove the disk currently left on the cutter.\n\n Press Next when complete.",
            image_path=[assets/"removeDisk.png"],
            next_target="finish",              # last page: Next becomes Finish
        )

        #Page 9: Display calibration result on completion
        self.pages["finish"] = InfoPage(
            self.container, self,
            title="END",
            body=finish_text,
            image_path=None,
            next_target=None,              # last page: Next becomes Finish
        )

        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_page(self, name):
        if self.current_page is not None and hasattr(self.current_page, "on_hide"):
            self.current_page.on_hide()
        page = self.pages[name]
        self.current_page = page
        self.logger.info(f"Showing page: {name}")
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()

    # --- shutdown --- #
    def stop_program(self):
        if self.stop_event.is_set():
            return
        self.logger.info("Stop requested")
        self.stop_event.set()
        self.after(25, self._wait_for_threads, 0)

    def _wait_for_threads(self, elapsed):
        with self._mqtt_lock:
            self._close_mqtt()
        with self._transport_lock:
            self._close_transport()
        alive = [t for t in self.threads if t.is_alive()]
        if alive and elapsed < timeout.THREAD_SHUTDOWN_TIMEOUT_MS:
            self.after(25, self._wait_for_threads, elapsed + 25)
            return
        if alive:
            self.logger.warning(f"{len(alive)} thread(s) did not stop in time; closing anyway.")
        self.logger.info("Closing wizard")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy()


# ===== Reusable widgets ===== #
def style_button(button, kind):
    colours = {
        "primary":  (gui.COLOR_ACCENT, gui.COLOR_ACCENT_ACTIVE, "white"),
        "danger":   (gui.COLOR_DANGER, gui.COLOR_DANGER_ACTIVE, "white"),
        "option":   (gui.COLOR_OPTION_BG, gui.COLOR_OPTION_ACTIVE, gui.COLOR_TEXT),
        "disabled": (gui.COLOR_DISABLED, gui.COLOR_DISABLED, "white"),
    }
    bg, active, fg = colours[kind]
    button.configure(bg=bg, activebackground=active, fg=fg, activeforeground=fg,
                     disabledforeground="white")


class ChoiceGroup(ttk.Frame):
    """A labelled row (or grid) of buttons where exactly one can be selected."""

    def __init__(self, parent, label, options, on_select, per_row=6):
        super().__init__(parent)
        self.value = None
        self._on_select = on_select
        self._buttons = {}
        self._enabled = True

        ttk.Label(self, text=label, font=gui.FONT_BODY_BOLD).pack(anchor="w", pady=(0, 6))
        grid = ttk.Frame(self)
        grid.pack(anchor="w")

        for i, (value, text) in enumerate(options):
            b = tk.Button(grid, text=text, font=gui.FONT_BUTTON, width=10, relief="flat",
                          bd=0, cursor="hand2", pady=6,
                          command=lambda v=value: self.select(v))
            b.grid(row=i // per_row, column=i % per_row, padx=4, pady=4)
            self._buttons[value] = b
        self._restyle()

    def select(self, value):
        self.value = value
        self._restyle()
        self._on_select(value)

    def clear(self):
        self.value = None
        self._restyle()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        for button in self._buttons.values():
            button.configure(state="normal" if enabled else "disabled",
                             cursor="hand2" if enabled else "arrow")
        self._restyle()

    def _restyle(self):
        for value, button in self._buttons.items():
            if not self._enabled:
                style_button(button, "disabled")
            else:
                style_button(button, "primary" if value == self.value else "option")


# ===== Pages ===== #
class BasePage(ttk.Frame):
    """Header (logo + title), content area, and Stop / Next footer."""

    def __init__(self, parent, app, title=""):
        super().__init__(parent)
        self.app = app

        footer = ttk.Frame(self, padding=(20, 15))
        footer.pack(side="bottom", fill="x")

        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side="top", fill="x")
        if app.logo_photo is not None:
            tk.Label(header, image=app.logo_photo, background=gui.COLOR_HEADER_BG).pack(
                side="left", padx=20, pady=12)
        if title:
            ttk.Label(header, text=title, style="HeaderTitle.TLabel").pack(side="left", padx=10)

        self.content = ttk.Frame(self, padding=20)
        self.content.pack(fill="both", expand=True)

        self.stop_button = tk.Button(footer, text="STOP PROGRAM", font=gui.FONT_BUTTON,
                                     relief="flat", bd=0, width=16, height=2,
                                     cursor="hand2", command=self.app.stop_program)
        style_button(self.stop_button, "danger")
        self.stop_button.pack(side="left", padx=5)

        self.next_button = tk.Button(footer, text="NEXT", font=gui.FONT_BUTTON,
                                     relief="flat", bd=0, width=12, height=2,
                                     cursor="hand2", command=self.on_next)
        style_button(self.next_button, "primary")
        self.next_button.pack(side="right", padx=5)

    def set_next_enabled(self, enabled: bool):
        self.next_button.configure(state="normal" if enabled else "disabled",
                                   cursor="hand2" if enabled else "arrow")
        style_button(self.next_button, "primary" if enabled else "disabled")

    def hide_next(self):
        self.next_button.pack_forget()

    def show_next(self):
        self.next_button.pack(side="right", padx=5)

    def on_next(self):
        raise NotImplementedError


class SelectionPage(BasePage):
    """Page 1: one question at a time.
    Factory -> Station -> Cluster (if the station has clusters) -> Cutter -> Transport.
    Changing an earlier answer removes the questions below it so they're asked again."""

    def __init__(self, parent, app, next_target):
        super().__init__(parent, app, title="Select Machine")
        self.next_target = next_target
        self._steps: list[ttk.Frame] = []     # question widgets currently shown, top to bottom
        self._transport_group = None

        body = ttk.Frame(self.content)
        body.pack(anchor="n", fill="x", padx=40)

        self.steps_frame = ttk.Frame(body)
        self.steps_frame.pack(anchor="w", fill="x")

        self.summary_label = ttk.Label(body, text="", font=gui.FONT_BODY)
        self.summary_label.pack(anchor="w", pady=(8, 0))

        factories = [(f, factory_name(f)) for f in config["FACTORIES"]]
        self._add_step("Factory", factories, self._on_factory)
        self._refresh()

    # --- building / clearing questions --- #
    def _add_step(self, label, options, on_select) -> ChoiceGroup:
        group = ChoiceGroup(self.steps_frame, label, options, on_select)
        group.pack(anchor="w", pady=(0, 16))
        self._steps.append(group)
        return group

    def _keep_steps(self, count):
        """Keep the first `count` questions and remove everything below them."""
        for widget in self._steps[count:]:
            if widget is self._transport_group:
                self._transport_group = None
            widget.destroy()
        del self._steps[count:]

    # --- answers --- #
    def _on_factory(self, value):
        sel = self.app.selection
        sel.factory = value
        sel.station = sel.cluster = sel.cutter = sel.transport_ok = None

        self._keep_steps(1)
        stations = [(s, f"Station {s}") for s in stations_for(value)]
        self._add_step("Station", stations, self._on_station)
        self._refresh()

    def _on_station(self, value):
        sel = self.app.selection
        sel.station = value
        sel.cluster = sel.cutter = sel.transport_ok = None

        self._keep_steps(2)
        layout = sel.layout()
        if "clusters" in layout:
            clusters = [(str(n), f"Cluster {n}") for n in range(1, layout["clusters"] + 1)]
            self._add_step("Cluster", clusters, self._on_cluster)
        else:
            self._add_cutter_step(cutter_count(layout))
        self._refresh()

    def _on_cluster(self, value):
        sel = self.app.selection
        sel.cluster = value
        sel.cutter = sel.transport_ok = None

        self._keep_steps(3)
        self._add_cutter_step(cutter_count(sel.layout()))
        self._refresh()

    def _add_cutter_step(self, count):
        cutters = [(str(n), f"Cutter {n}") for n in range(1, count + 1)]
        self._add_step("Cutter", cutters, self._on_cutter)

    def _on_cutter(self, value):
        self.app.selection.cutter = value
        # Changing cutter keeps the transport answer; it's only asked once per location
        if self._transport_group is None:
            self._transport_group = self._add_step(
                "Is transport operational?", [(True, "Yes"), (False, "No")], self._on_transport)
        self._refresh()

    def _on_transport(self, value):
        self.app.selection.transport_ok = value
        self._refresh()

    def _refresh(self):
        self.summary_label.configure(text=self.app.selection.summary())
        self.set_next_enabled(self.app.selection.is_complete())

    def on_next(self):
        if not self.app.selection.is_complete():
            return
        sel = self.app.selection
        sel.log_file, sel.log_folder = start_file_logging(self.app.logger, sel.machine_id)
        self.app.logger.info(f"Machine ID: {sel.machine_id}")
        self.app.logger.info(f"Selection: {sel.summary()}")
        self.app.show_page(self.next_target)


class InfoPage(BasePage):
    """Text plus optional images that resize to fit the window."""

    MIN_IMAGE_HEIGHT = 120

    def __init__(self, parent, app, title, body, next_target, image_path=None):
        super().__init__(parent, app, title=title)
        self.next_target = next_target
        self._pil_images, self._photos, self.image_labels = [], [], []
        self._resize_job = None
        

        container = ttk.Frame(self.content)
        container.pack(expand=True, fill="both")

        self.body_source = body
        self.body_label = ttk.Label(container, text="" if callable(body) else body,
                                            font=gui.FONT_BODY, justify="left", wraplength=900)
        self.body_label.pack(pady=(0, 15))

        if image_path:
            paths = image_path if isinstance(image_path, (list, tuple)) else [image_path]
            self.images_frame = ttk.Frame(container)
            self.images_frame.pack(pady=10, fill="both", expand=True)
            for path in paths:
                label = ttk.Label(self.images_frame)
                label.pack(side="left", padx=10, expand=True)
                self.image_labels.append(label)
                try:
                    img = Image.open(path)
                    img.load()
                    self._pil_images.append(img)
                except Exception:
                    self.app.logger.exception(f"Could not load image {path}")
                    self._pil_images.append(None)
                    label.configure(text=f"[image not found: {Path(path).name}]")
            self.bind("<Configure>", self._on_configure)

        if next_target is None:
            self.next_button.configure(text="FINISH", command=self.app.stop_program)

    def _on_configure(self, _event):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._resize_images)

    def _resize_images(self):
        self._resize_job = None
        self.update_idletasks()
        available_h = max(self.content.winfo_height() - self.body_label.winfo_height() - 60,
                          self.MIN_IMAGE_HEIGHT)
        available_w = self.images_frame.winfo_width() or self.content.winfo_width()
        per_image_w = max(available_w // len(self._pil_images), 100)

        self._photos = []
        for label, pil_img in zip(self.image_labels, self._pil_images):
            if pil_img is None:
                continue
            img = pil_img.copy()
            img.thumbnail((per_image_w, available_h))
            photo = ImageTk.PhotoImage(img)
            self._photos.append(photo)
            label.configure(image=photo, text="")
    
    def on_show(self):
        if callable(self.body_source):
            try:
                text = self.body_source(self.app.make_context())
            except Exception:
                self.app.logger.exception("Could not build page text")
                text = "Could not load instructions - check the log."
            self.body_label.configure(text=text)

    def on_next(self):
        if self.next_target:
            self.app.show_page(self.next_target)


class ConfirmPage(BasePage):
    """Text plus a Yes / No question. Next goes to yes_target or no_target.
    If no_target is None, answering No keeps Next locked and shows no_message."""

    def __init__(self, parent, app, title, body, question, key, yes_target,
                 no_target=None, no_message="Select Yes to continue."):
        super().__init__(parent, app, title=title)
        self.key = key
        self.yes_target = yes_target
        self.no_target = no_target
        self.no_message = no_message

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        ttk.Label(center, text=body, font=gui.FONT_BODY, justify="left").pack(anchor="w", pady=(0, 24))
        ChoiceGroup(center, question, [(True, "Yes"), (False, "No")],
                    self._on_answer).pack(anchor="w")

        self.message_label = tk.Label(center, text="", font=gui.FONT_BODY_BOLD, bg=gui.COLOR_BG,
                                      fg=gui.COLOR_DANGER)
        self.message_label.pack(anchor="w", pady=(16, 0))

        self.set_next_enabled(False)

    def _on_answer(self, value):
        self.app.answers[self.key] = value
        self.app.logger.info(f"Answer '{self.key}': {'Yes' if value else 'No'}")
        blocked = not value and self.no_target is None
        self.message_label.configure(text=self.no_message if blocked else "")
        self.set_next_enabled(not blocked)

    def on_next(self):
        answer = self.app.answers.get(self.key)
        target = self.yes_target if answer else self.no_target
        if target:
            self.app.show_page(target)

class CheckPage(BasePage):
    """Runs a check function (returns True / False) when the page opens and each
    time RELOAD is pressed. True unlocks Next; False shows fail_message."""

    def __init__(self, parent, app, title, body, check, next_target,
                 pass_message="OK. Press Next to continue.",
                 fail_message="Check failed. Fix the problem and press Reload."):
        super().__init__(parent, app, title=title)
        self.check = check
        self.next_target = next_target
        self.pass_message = pass_message
        self.fail_message = fail_message
        self._passed = False
        self._request_id = 0   # results from older checks are ignored

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        ttk.Label(center, text=body, font=gui.FONT_BODY, justify="center").pack(pady=(0, 24))

        self.status_label = tk.Label(center, text="", font=gui.FONT_BODY_BOLD, bg=gui.COLOR_BG,
                                     fg=gui.COLOR_TEXT, justify="center", wraplength=700)
        self.status_label.pack(pady=(0, 20))

        self.reload_button = tk.Button(center, text="RELOAD", font=gui.FONT_BUTTON, relief="flat",
                                       bd=0, width=16, height=2, cursor="hand2",
                                       command=self.reload)
        style_button(self.reload_button, "primary")
        self.reload_button.pack()

        self.set_next_enabled(False)

    def on_show(self):
        self.reload()

    def on_hide(self):
        self._request_id += 1  # drop any check still in flight

    def reload(self):
        self._request_id += 1
        self._passed = False
        self.set_next_enabled(False)
        self.reload_button.configure(state="disabled", cursor="arrow")
        style_button(self.reload_button, "disabled")
        self.status_label.configure(text="Checking...", fg=gui.COLOR_TEXT)
        self.app.run_in_background(self._run_check, self._request_id)

    def _run_check(self, request_id):
        try:
            ok = bool(self.check(self.app.make_context()))
            self.app.call_in_ui(self._show_result, request_id, ok, None)
        except Exception as e:
            self.app.logger.exception(f"{self.check.__name__} failed")
            self.app.call_in_ui(self._show_result, request_id, False, str(e))

    def _show_result(self, request_id, ok, error):
        if request_id != self._request_id or self.app.stop_event.is_set():
            return  # an older check, or the page has been left
        self.reload_button.configure(state="normal", cursor="hand2")
        style_button(self.reload_button, "primary")

        if error is not None:
            self.status_label.configure(text=f"Check could not run: {error}", fg=gui.COLOR_DANGER)
        elif ok:
            self.status_label.configure(text=self.pass_message, fg=gui.COLOR_SUCCESS)
        else:
            self.status_label.configure(text=self.fail_message, fg=gui.COLOR_DANGER)

        self._passed = ok and error is None
        self.set_next_enabled(self._passed)

    def on_next(self):
        if self._passed:
            self.app.show_page(self.next_target)

class ImageTaskPage(BasePage):
    """Shows an image and runs a task as soon as the page appears.
    Moves on automatically when the task succeeds; shows RETRY if it fails.
    The image resizes to fit the space above the status area."""

    MIN_IMAGE_HEIGHT = 120

    def __init__(self, parent, app, title, body, task, next_target, image_path=None):
        super().__init__(parent, app, title=title)
        self.task = task
        self.next_target = next_target
        self._running = False
        self._pil_image = None
        self._photo = None
        self._resize_job = None

        container = ttk.Frame(self.content)
        container.pack(expand=True, fill="both")

        ttk.Label(container, text=body, font=gui.FONT_BODY, justify="center").pack(pady=(0, 10))

        # Status area at the bottom, packed first so the image gets the remaining space
        status_frame = ttk.Frame(container)
        status_frame.pack(side="bottom", pady=(10, 0))
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=400)
        self.progress.pack(pady=5)
        self.status_label = tk.Label(status_frame, text="", font=gui.FONT_BODY_BOLD,
                                     bg=gui.COLOR_BG, fg=gui.COLOR_TEXT, wraplength=700)
        self.status_label.pack(pady=5)
        self.retry_button = tk.Button(status_frame, text="RETRY", font=gui.FONT_BUTTON,
                                      relief="flat", bd=0, width=16, height=2,
                                      cursor="hand2", command=self._start)
        style_button(self.retry_button, "primary")

        # Image fills the space between the text and the status area
        self.image_label = ttk.Label(container, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        # A path, or a function (ctx) -> path, looked up when the page is shown
        self.image_source = image_path
        self.image_label.bind("<Configure>", self._on_configure)

        # Moves on by itself, so no Next button
        self.hide_next()

    # --- image sizing --- #
    def _load_image(self):
        """Resolve image_source and display it. Safe to call when there's no image yet."""
        source = self.image_source
        if callable(source):
            try:
                source = source(self.app.make_context())
            except Exception:
                self.app.logger.exception("Could not find image")
                source = None
        if not source:
            return
        try:
            img = Image.open(source)
            img.load()
        except Exception:
            self.app.logger.exception(f"Could not load image {source}")
            self.image_label.configure(text=f"[could not load image: {Path(source).name}]")
            return
        self._pil_image = img
        self._resize_image()

    def _on_configure(self, _event):
        if self._pil_image is None:
            return
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._resize_image)

    def _resize_image(self):
        self._resize_job = None
        w = max(self.image_label.winfo_width(), 100)
        h = max(self.image_label.winfo_height(), self.MIN_IMAGE_HEIGHT)
        img = self._pil_image.copy()
        img.thumbnail((w, h))
        self._photo = ImageTk.PhotoImage(img)   # keep a reference or Tkinter drops it
        self.image_label.configure(image=self._photo, text="")

    # --- running the task --- #
    def on_show(self):
        self._load_image()
        if not self._running:
            self._start()

    def _start(self):
        self._running = True
        self.retry_button.pack_forget()
        self.status_label.configure(text="Running...", fg=gui.COLOR_TEXT)
        self.progress.start(10)
        self.app.run_in_background(self._run)

    def _run(self):
        try:
            self.task(self.app.make_context())
            self.app.call_in_ui(self._finished, True, None)
        except Exception as e:
            self.app.logger.exception(f"{self.task.__name__} failed")
            self.app.call_in_ui(self._finished, False, str(e))

    def _finished(self, ok, error):
        self._running = False
        self.progress.stop()
        if self.app.stop_event.is_set():
            return
        if ok:
            self.status_label.configure(text="Complete. Moving on...", fg=gui.COLOR_SUCCESS)
            self._load_image()
            self.after(timeout.AUTO_ADVANCE_DELAY_MS, self.app.show_page, self.next_target)
        else:
            self.status_label.configure(text=f"Failed: {error}", fg=gui.COLOR_DANGER)
            self.retry_button.pack(pady=5)

    def on_next(self):
        pass  # no Next button; the page moves on by itself

class ReloadConfirmPage(BasePage):
    """Reads a value (e.g. the diamond ID from Airtable) when the page opens and each
    time RELOAD is pressed, then asks the operator to confirm it with Yes / No.
    Yes unlocks Next. No tells them to fill out the form and reload.
    Reloading clears the previous answer so the new value has to be confirmed.
    If decide_next is given, it runs in the background when Next is pressed and
    returns the name of the page to go to; otherwise Next goes to next_target."""

    def __init__(self, parent, app, title, body, value_label, read_value, value_key,
                 question, key, next_target, no_message, form_url=None, decide_next=None):
        super().__init__(parent, app, title=title)
        self.read_value = read_value
        self.value_key = value_key
        self.key = key
        self.next_target = next_target
        self.no_message = no_message
        self.form_url = form_url
        self.decide_next = decide_next
        self._request_id = 0   # results from older reads are ignored
        self._value_text = None

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        ttk.Label(center, text=body, font=gui.FONT_BODY, justify="left").pack(anchor="w", pady=(0, 16))

        card = tk.Frame(center, bg=gui.COLOR_CARD_BG, padx=60, pady=24)
        card.pack(anchor="w", fill="x")
        tk.Label(card, text=value_label, font=gui.FONT_BODY_BOLD, bg=gui.COLOR_CARD_BG,
                 fg=gui.COLOR_TEXT).pack()
        self.value_display = tk.Label(card, text="--", font=FONT_LIVE_VALUE, bg=gui.COLOR_CARD_BG,
                                      fg=gui.COLOR_TEXT)
        self.value_display.pack(pady=(8, 8))
        self.updated_label = tk.Label(card, text="", font=gui.FONT_BODY,
                                      bg=gui.COLOR_CARD_BG, fg=gui.COLOR_DISABLED)
        self.updated_label.pack()

        self.reload_button = tk.Button(center, text="RELOAD", font=gui.FONT_BUTTON, relief="flat",
                                       bd=0, width=16, height=2, cursor="hand2",
                                       command=self.reload)
        style_button(self.reload_button, "primary")
        self.reload_button.pack(anchor="w", pady=(16, 24))

        self.choices = ChoiceGroup(center, question, [(True, "Yes"), (False, "No")],
                                   self._on_answer)
        self.choices.pack(anchor="w")

        self.message_label = tk.Label(center, text="", font=gui.FONT_BODY_BOLD, bg=gui.COLOR_BG,
                                      fg=gui.COLOR_DANGER, justify="left")
        self.message_label.pack(anchor="w", pady=(16, 0))

        # Only shown after "No", and only if a form URL is configured
        self.form_button = tk.Button(center, text="OPEN FORM", font=gui.FONT_BUTTON, relief="flat",
                                     bd=0, width=16, height=2, cursor="hand2",
                                     command=self._open_form)
        style_button(self.form_button, "primary")

        self.set_next_enabled(False)

    def on_show(self):
        self.reload()

    def on_hide(self):
        self._request_id += 1  # drop any read or check still in flight

    # --- reading the value --- #
    def reload(self):
        self._request_id += 1
        # Previous answer no longer applies to whatever value comes back
        self.app.answers.pop(self.key, None)
        self.choices.clear()
        self.choices.set_enabled(False)
        self._show_message(False)
        self.set_next_enabled(False)

        self.reload_button.configure(state="disabled", cursor="arrow")
        style_button(self.reload_button, "disabled")
        self.value_display.configure(fg=gui.COLOR_DISABLED)
        self.updated_label.configure(text="Reading...", fg=gui.COLOR_DISABLED)
        self.app.run_in_background(self._read, self._request_id)

    def _read(self, request_id):
        try:
            value = self.read_value(self.app.make_context())
            self.app.call_in_ui(self._show_value, request_id, value, None)
        except Exception as e:
            self.app.logger.exception(f"Reading {self.value_key} failed")
            self.app.call_in_ui(self._show_value, request_id, None, str(e))

    def _show_value(self, request_id, value, error):
        if request_id != self._request_id or self.app.stop_event.is_set():
            return  # an older read, or the page has been left
        self.reload_button.configure(state="normal", cursor="hand2")
        style_button(self.reload_button, "primary")

        if error is not None or value in (None, ""):
            self.app.values.pop(self.value_key, None)
            self.value_display.configure(text="--", fg=gui.COLOR_DANGER)
            reason = error or "no value found"
            self.updated_label.configure(text=f"Read failed: {reason}", fg=gui.COLOR_DANGER)
            return

        self._value_text = str(value)
        self.app.values[self.value_key] = value
        self.value_display.configure(text=self._value_text, fg=gui.COLOR_TEXT)
        self.updated_label.configure(text=f"Read at {time.strftime('%H:%M:%S')}",
                                     fg=gui.COLOR_DISABLED)
        self.app.logger.info(f"Read {self.value_key}: {self._value_text}")
        self.choices.set_enabled(True)

    # --- Yes / No --- #
    def _on_answer(self, value):
        self.app.answers[self.key] = value
        self.app.logger.info(f"{self.value_key} {self._value_text} confirmed: "
                             f"{'Yes' if value else 'No'}")
        self._show_message(not value)
        self.set_next_enabled(value)

    def _show_message(self, show: bool):
        self.message_label.configure(text=self.no_message if show else "", fg=gui.COLOR_DANGER)
        if show and self.form_url:
            self.form_button.pack(anchor="w", pady=(12, 0))
        else:
            self.form_button.pack_forget()

    def _open_form(self):
        self.app.logger.info(f"Opening form: {self.form_url}")
        webbrowser.open(self.form_url)

    # --- moving on --- #
    def on_next(self):
        if not self.app.answers.get(self.key):
            return
        if self.decide_next is None:
            self.app.show_page(self.next_target)
            return

        # Lock the page while the check runs
        self._request_id += 1
        self.set_next_enabled(False)
        self.choices.set_enabled(False)
        self.reload_button.configure(state="disabled", cursor="arrow")
        style_button(self.reload_button, "disabled")
        self.message_label.configure(text="Checking...", fg=gui.COLOR_TEXT)
        self.app.run_in_background(self._decide, self._request_id)

    def _decide(self, request_id):
        try:
            target = self.decide_next(self.app.make_context())
            self.app.call_in_ui(self._decided, request_id, target, None)
        except Exception as e:
            self.app.logger.exception(f"{self.decide_next.__name__} failed")
            self.app.call_in_ui(self._decided, request_id, None, str(e))

    def _decided(self, request_id, target, error):
        if request_id != self._request_id or self.app.stop_event.is_set():
            return
        self.choices.set_enabled(True)   # keeps the existing Yes selection
        self.reload_button.configure(state="normal", cursor="hand2")
        style_button(self.reload_button, "primary")

        if error is None and target not in self.app.pages:
            error = f"unknown page '{target}'"
        if error is not None:
            # Leave Next enabled so pressing it again retries the check
            self.message_label.configure(text=f"Check failed: {error}", fg=gui.COLOR_DANGER)
            self.set_next_enabled(True)
            return

        self.message_label.configure(text="")
        self.app.logger.info(f"{self.decide_next.__name__} -> {target}")
        self.app.show_page(target)


class ReloadValuePage(BasePage):
    """Shows a value that is read once when the page opens and again each time
    RELOAD is pressed. read_value runs in a background thread, so a slow or
    blocking read (e.g. an MQTT subscribe) can't freeze the window.
    Next stays locked until a read has succeeded."""

    def __init__(self, parent, app, title, body, value_label, read_value, next_target,
                 unit="", decimals=2, valid_range=None):
        super().__init__(parent, app, title=title)
        self.read_value = read_value
        self.next_target = next_target
        self.unit = unit
        self.decimals = decimals
        self.valid_range = valid_range   # (min, max) or None for no check
        self._value_ok = False
        self._request_id = 0

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        ttk.Label(center, text=body, font=gui.FONT_BODY, justify="center").pack(pady=(0, 24))

        card = tk.Frame(center, bg=gui.COLOR_CARD_BG, padx=60, pady=30)
        card.pack()
        tk.Label(card, text=value_label, font=gui.FONT_BODY_BOLD, bg=gui.COLOR_CARD_BG,
                 fg=gui.COLOR_TEXT).pack()
        self.value_display = tk.Label(card, text="--", font=FONT_LIVE_VALUE, bg=gui.COLOR_CARD_BG,
                                      fg=gui.COLOR_TEXT, width=10)
        self.value_display.pack(pady=(8, 8))
        self.updated_label = tk.Label(card, text="", font=gui.FONT_BODY,
                                      bg=gui.COLOR_CARD_BG, fg=gui.COLOR_DISABLED)
        self.updated_label.pack()

        self.reload_button = tk.Button(center, text="RELOAD", font=gui.FONT_BUTTON, relief="flat",
                                       bd=0, width=16, height=2, cursor="hand2",
                                       command=self.reload)
        style_button(self.reload_button, "primary")
        self.reload_button.pack(pady=(20, 0))

        self.set_next_enabled(False)

    def on_show(self):
        self.reload()

    def on_hide(self):
        self._request_id += 1  # drop any read still in flight

    def reload(self):
        self._request_id += 1
        self.reload_button.configure(state="disabled", cursor="arrow")
        style_button(self.reload_button, "disabled")
        self.set_next_enabled(False)
        self.updated_label.configure(text="Reading...", fg=gui.COLOR_DISABLED)
        self.app.run_in_background(self._read, self._request_id)

    def _read(self, request_id):
        try:
            value = self.read_value(self.app.make_context())
            self.app.call_in_ui(self._show_value, request_id, value, None)
        except Exception as e:
            self.app.logger.exception("Value read failed")
            self.app.call_in_ui(self._show_value, request_id, None, str(e))

    def _show_value(self, request_id, value, error):
        if request_id != self._request_id or self.app.stop_event.is_set():
            return  # an older read, or the page has been left
        self.reload_button.configure(state="normal", cursor="hand2")
        style_button(self.reload_button, "primary")

        if error is not None:
            self.value_display.configure(text="--", fg=gui.COLOR_DANGER)
            self.updated_label.configure(text=f"Read failed: {error}", fg=gui.COLOR_DANGER)
            return

        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None

        text = f"{number:.{self.decimals}f}{self.unit}" if number is not None else f"{value}{self.unit}"
        read_time = time.strftime('%H:%M:%S')

        if self.valid_range is None:
            ok, message = True, f"Read at {read_time}"
        elif number is None:
            ok, message = False, f"Not a number (read at {read_time})"
        else:
            lo, hi = self.valid_range
            ok = lo <= number <= hi
            message = (f"In range. Read at {read_time}" if ok else
                       f"Out of range ({lo} - {hi}{self.unit}). Adjust and press Reload.")

        self._value_ok = ok
        self.value_display.configure(text=text, fg=gui.COLOR_SUCCESS if ok else gui.COLOR_DANGER)
        self.updated_label.configure(text=message, fg=gui.COLOR_DISABLED if ok else gui.COLOR_DANGER)
        self.app.logger.info(f"Read value: {text} ({'in range' if ok else 'out of range'})")
        self.set_next_enabled(ok)

    def on_next(self):
        if self._value_ok:
            self.app.show_page(self.next_target)


class TaskPage(BasePage):
    """If transport is operational, runs its task as soon as the page is shown and
    moves on automatically when it succeeds (RETRY on failure).
    If transport is not operational, skips the task and shows a manual instruction
    for the current disk_position, with Next for the operator to press when done.

    The task can pause for the operator with ctx.ask_operator(message), which shows
    the message with a CONFIRM button and waits until it's pressed."""

    def __init__(self, parent, app, title, body, task, next_target, manual_instructions,
                 manual_next_position=None):
        """next_target: a page name, or a function (ctx) -> page name.
        manual_instructions: {"No Disk": "...", "A_side": "...", "B_side": "..."}
        manual_next_position: {disk_position: new disk_position} applied when the operator
            presses Next in manual mode, or None to leave disk_position unchanged."""
        super().__init__(parent, app, title=title)
        self.task = task
        self.next_target = next_target
        self.auto_body = body
        self.manual_instructions = manual_instructions
        self.manual_next_position = manual_next_position
        self._running = False
        self._confirm_event = None

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        self.body_label = ttk.Label(center, text=body, font=gui.FONT_BODY, justify="center")
        self.body_label.pack(pady=(0, 20))

        # Automatic mode widgets
        self.auto_frame = ttk.Frame(center)
        self.progress = ttk.Progressbar(self.auto_frame, mode="indeterminate", length=400)
        self.progress.pack(pady=10)
        self.status_label = tk.Label(self.auto_frame, text="", font=gui.FONT_BODY_BOLD,
                                     bg=gui.COLOR_BG, fg=gui.COLOR_TEXT, justify="center", wraplength=700)
        self.status_label.pack(pady=10)
        self.retry_button = tk.Button(self.auto_frame, text="RETRY", font=gui.FONT_BUTTON,
                                      relief="flat", bd=0, width=16, height=2,
                                      cursor="hand2", command=self._start)
        style_button(self.retry_button, "primary")
        # Only shown while the task is waiting for the operator (ctx.ask_operator)
        self.confirm_button = tk.Button(self.auto_frame, text="CONFIRM", font=gui.FONT_BUTTON,
                                        relief="flat", bd=0, width=16, height=2,
                                        cursor="hand2", command=self._on_confirm)
        style_button(self.confirm_button, "primary")

        self.hide_next()

    def on_show(self):
        if self.app.selection.transport_ok:
            self._show_auto()
        else:
            self._show_manual()

    def _resolve_next(self, ctx):
        """next_target can be a page name or a function (ctx) -> page name."""
        target = self.next_target(ctx) if callable(self.next_target) else self.next_target
        if target not in self.app.pages:
            raise RuntimeError(f"unknown page '{target}'")
        return target

    # --- transport not operational: manual instruction --- #
    def _show_manual(self):
        self.auto_frame.pack_forget()
        instruction = self.manual_instructions.get(
            disk_position, f"No instruction set for disk position '{disk_position}'")
        self.body_label.configure(text=instruction)
        self.app.logger.info(f"Transport not operational: showing manual instruction "
                             f"({disk_position})")
        self.set_next_enabled(True)
        self.show_next()

    # --- transport operational: run the task --- #
    def _show_auto(self):
        self.hide_next()
        self.body_label.configure(text=self.auto_body)
        self.auto_frame.pack()
        if not self._running:
            self._start()

    def _start(self):
        self._running = True
        self.retry_button.pack_forget()
        self.confirm_button.pack_forget()
        self.status_label.configure(text="Running...", fg=gui.COLOR_TEXT)
        self.progress.start(10)
        self.app.run_in_background(self._run)

    def _run(self):
        ctx = self.app.make_context()
        ctx.ask_operator = lambda message: self.ask_operator(message, ctx.stop_event)
        try:
            self.task(ctx)
            target = self._resolve_next(ctx)
            self.app.call_in_ui(self._finished, True, None, target)
        except Exception as e:
            self.app.logger.exception(f"{self.task.__name__} failed")
            self.app.call_in_ui(self._finished, False, str(e), None)

    def _finished(self, ok, error, target):
        self._running = False
        self.progress.stop()
        self.confirm_button.pack_forget()
        self._confirm_event = None
        if self.app.stop_event.is_set():
            return
        if ok:
            self.status_label.configure(text="Complete. Moving on...", fg=gui.COLOR_SUCCESS)
            self.after(AUTO_ADVANCE_DELAY_MS, self.app.show_page, target)
        else:
            self.status_label.configure(text=f"Failed: {error}", fg=gui.COLOR_DANGER)
            self.retry_button.pack(pady=10)

    # --- asking the operator from inside the task --- #
    def ask_operator(self, message, stop_event):
        """Called from the task thread. Shows message with a CONFIRM button and blocks
        until the operator presses it. Raises if STOP PROGRAM is pressed while waiting."""
        done = threading.Event()
        self.app.logger.info(f"Waiting for operator: {message}")
        self.app.call_in_ui(self._show_prompt, message, done)
        while not done.wait(0.2):
            if stop_event.is_set():
                raise RuntimeError("Stopped while waiting for the operator")
        self.app.logger.info("Operator confirmed")

    def _show_prompt(self, message, done):
        self._confirm_event = done
        self.progress.stop()
        self.status_label.configure(text=message, fg=gui.COLOR_TEXT)
        self.confirm_button.pack(pady=10)

    def _on_confirm(self):
        self.confirm_button.pack_forget()
        self.status_label.configure(text="Running...", fg=gui.COLOR_TEXT)
        self.progress.start(10)
        if self._confirm_event is not None:
            self._confirm_event.set()
            self._confirm_event = None

    def on_next(self):
        # Only reachable in manual mode
        global disk_position
        try:
            target = self._resolve_next(self.app.make_context())
        except Exception as e:
            self.app.logger.exception("Could not decide the next page")
            self.body_label.configure(text=f"Could not decide the next page: {e}")
            return
        if self.manual_next_position:
            new_position = self.manual_next_position.get(disk_position, disk_position)
            if new_position != disk_position:
                self.app.logger.info(f"Disk position {disk_position} -> {new_position} (manual)")
                disk_position = new_position
        self.app.show_page(target)


class CameraLogPage(BasePage):
    """Live camera feed on the left, live logs on the right.
    Starts its worker when shown and moves on automatically when it succeeds."""

    def __init__(self, parent, app, title, worker, next_target):
        super().__init__(parent, app, title=title)
        self.worker = worker
        self.next_target = next_target
        self.frames = FrameBuffer()
        self.log_queue: queue.Queue = queue.Queue()
        self._handler = None
        self._running = False
        self._started = False
        self.tk_image = None

        camera_frame = ttk.LabelFrame(self.content, text="Camera Feed", padding=10)
        camera_frame.pack(side="left", fill="both", padx=(0, 10))
        camera_frame.configure(width=gui.CAMERA_DISPLAY_SIZE[0] + 40,
                               height=gui.CAMERA_DISPLAY_SIZE[1] + 40)
        camera_frame.pack_propagate(False)
        self.camera_label = ttk.Label(camera_frame, text="Waiting for camera...",
                                      style="Card.TLabel", anchor="center")
        self.camera_label.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(self.content, text="Live Log", padding=10)
        log_frame.pack(side="right", fill="both", expand=True)
        self.status_label = tk.Label(log_frame, text="Status: Waiting", font=gui.FONT_BODY_BOLD,
                                     bg=gui.COLOR_CARD_BG, fg=gui.COLOR_TEXT, anchor="w")
        self.status_label.pack(fill="x", pady=(0, 8))
        self.log_text = ScrolledText(log_frame, font=gui.FONT_LOG, wrap="word", state="disabled",
                                     relief="flat", bd=0)
        self.log_text.pack(fill="both", expand=True)

        # Moves on by itself; Next only reappears (as RETRY) if the run fails
        self.hide_next()
    
    def on_show(self):
        if not self._running:
            self._start()

    def _start(self):
        self._started = True
        self._running = True
        self.frames.clear()
        self.hide_next()
        self.status_label.configure(text="Status: Running", fg=gui.COLOR_TEXT)

        self._handler = QueueLogHandler(self.log_queue)
        self.app.logger.addHandler(self._handler)

        self.app.run_in_background(self._run)
        self._update_camera()
        self._drain_logs()

    def _run(self):
        try:
            self.worker(self.app.make_context(self.frames))
            self.app.call_in_ui(self._finished, True, None)
        except Exception as e:
            self.app.logger.exception(f"{self.worker.__name__} failed")
            self.app.call_in_ui(self._finished, False, str(e))

    def _finished(self, ok, error):
        self._running = False
        self._drain_logs()  # flush anything still queued
        if self._handler is not None:
            self.app.logger.removeHandler(self._handler)
            self._handler = None
        if self.app.stop_event.is_set():
            return
        if ok:
            try:
                target = self._resolve_next()
            except Exception as e:
                self.app.logger.exception("Could not decide the next page")
                self.status_label.configure(text=f"Status: Failed: {e}", fg=gui.COLOR_DANGER)
                return
            self.status_label.configure(text="Status: Complete. Moving on...", fg=gui.COLOR_SUCCESS)
            self.after(AUTO_ADVANCE_DELAY_MS, self.app.show_page, target)

    def _update_camera(self):
        if not self._running or self.app.stop_event.is_set():
            return
        frame = self.frames.get()
        if frame is not None:
            try:
                if frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)   # grayscale -> 3 channels
                else:
                    rgb = cv2.cvtColor(frame, cv2.gui.COLOR_BGR2RGB)    # OpenCV BGR -> Tkinter RGB
                image = Image.fromarray(rgb)
                image.thumbnail(gui.CAMERA_DISPLAY_SIZE)
                self.tk_image = ImageTk.PhotoImage(image)
                self.camera_label.configure(image=self.tk_image, text="")
            except Exception:
                self.app.logger.exception("Error updating camera frame")
        self.after(33, self._update_camera)

    def _drain_logs(self):
        lines = []
        try:
            while True:
                lines.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if lines:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", "\n".join(lines) + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        if self._running and not self.app.stop_event.is_set():
            self.after(100, self._drain_logs)
    
    def _resolve_next(self):
        """next_target can be a page name or a function (ctx) -> page name."""
        if callable(self.next_target):
            return self.next_target(self.app.make_context())
        return self.next_target

    def on_next(self):
        pass  # replaced by RETRY on failure



class RepeatCheckPage(BasePage):
    """Shows instructions built from ctx, with an action button (e.g. OK).
    The button runs action(ctx) in the background:
      True  -> pass_message is shown and Next unlocks
      False -> the instructions are rebuilt from the new values and the button
               is available again, so the operator can repeat the step."""

    def __init__(self, parent, app, title, body, action, next_target, button_text="OK",
                 pass_message="Complete. Press Next to continue.",
                 fail_message="Still not within range. Follow the updated instructions "
                              "and press OK again."):
        super().__init__(parent, app, title=title)
        self.body_source = body
        self.action = action
        self.next_target = next_target
        self.pass_message = pass_message
        self.fail_message = fail_message
        self._passed = False
        self._running = False

        center = ttk.Frame(self.content)
        center.pack(expand=True)

        self.body_label = ttk.Label(center, text="", font=gui.FONT_BODY, justify="center",
                                    wraplength=800)
        self.body_label.pack(pady=(0, 20))

        self.action_button = tk.Button(center, text=button_text, font=gui.FONT_BUTTON,
                                       relief="flat", bd=0, width=16, height=2,
                                       cursor="hand2", command=self._start)
        style_button(self.action_button, "primary")
        self.action_button.pack(pady=10)

        self.progress = ttk.Progressbar(center, mode="indeterminate", length=400)
        self.progress.pack(pady=10)

        self.status_label = tk.Label(center, text="", font=gui.FONT_BODY_BOLD, bg=gui.COLOR_BG,
                                     fg=gui.COLOR_TEXT, justify="center", wraplength=700)
        self.status_label.pack(pady=10)

        self.set_next_enabled(False)

    def on_show(self):
        if self._running:
            return
        self._passed = False
        self.set_next_enabled(False)
        self.status_label.configure(text="")
        self._set_button(True)
        self._refresh_body()

    def _refresh_body(self):
        try:
            text = self.body_source(self.app.make_context())
        except Exception:
            self.app.logger.exception("Could not build page text")
            text = "Could not load instructions - check the log."
        self.body_label.configure(text=text)

    def _set_button(self, enabled):
        self.action_button.configure(state="normal" if enabled else "disabled",
                                     cursor="hand2" if enabled else "arrow")
        style_button(self.action_button, "primary" if enabled else "disabled")

    def _start(self):
        if self._running:
            return
        self._running = True
        self._set_button(False)
        self.set_next_enabled(False)
        self.status_label.configure(text="Measuring...", fg=gui.COLOR_TEXT)
        self.progress.start(10)
        self.app.run_in_background(self._run)

    def _run(self):
        try:
            ok = bool(self.action(self.app.make_context()))
            self.app.call_in_ui(self._finished, ok, None)
        except Exception as e:
            self.app.logger.exception(f"{self.action.__name__} failed")
            self.app.call_in_ui(self._finished, False, str(e))

    def _finished(self, ok, error):
        self._running = False
        self.progress.stop()
        if self.app.stop_event.is_set():
            return
        if error is not None:
            self.status_label.configure(text=f"Failed: {error}\n\nPress OK to try again.",
                                        fg=gui.COLOR_DANGER)
            self._set_button(True)
        elif ok:
            self._passed = True
            self.status_label.configure(text=self.pass_message, fg=gui.COLOR_SUCCESS)
            self.set_next_enabled(True)        # OK stays disabled; the step is done
        else:
            self._refresh_body()               # show the new readings
            self.status_label.configure(text=self.fail_message, fg=gui.COLOR_DANGER)
            self._set_button(True)

    def on_next(self):
        if self._passed:
            self.app.show_page(self.next_target)

# ===== Tasks and workers: replace these with the real processes ===== #
# Each receives a WorkerContext:
#   ctx.selection   the choices from page 1 (factory, station, cluster or None, cutter, transport_ok)
#   ctx.selection.machine_id   "station.cluster.cutter", e.g. "2.3.1"
#   ctx.answers     Yes/No answers from confirm pages, e.g. ctx.answers["ready"]
#   ctx.values      values read on pages, e.g. ctx.values["diamond_id"]
#   ctx.stop_event  set when STOP PROGRAM is pressed; check it and return early
#   ctx.log(msg)    shows up in the live log (so does anything logged via the app logger)
#   ctx.frames.set(frame)  publish a BGR numpy frame to the camera panel
#   ctx.ask_operator(msg)  TaskPage tasks only: show msg + CONFIRM and wait for the operator
# Raise an exception to mark the step as failed.

def read_diamond_id(ctx: WorkerContext):
    """Return the diamond ID for the selected cutter by pulling data from airtable.
    Called when the Diamond ID page opens and each time Reload is pressed."""
    machine_id = ctx.selection.machine_id     # e.g. "2.3.1"
    ctx.log(f"Looking up diamond ID for {machine_id}")
    diamond = airtable.get_diamond_ID_from_location(machine_id)
    if diamond == None:
        return "No diamond recorded in airtable please log diamond and reload"
    else:
        return diamond

def read_cutter_state(ctx: WorkerContext):
    """Returns True if the cutter is out of production and in IDLE."""
    api = ctx.get_api()
    mode = int(api.get("packml_mode"))
    state = int(api.get("packml_state"))
    ctx.log(f"PackML mode {mode}, state {state}")

    if mode == 1:
        ctx.log("Cutter is in production, not continuing with calibration")
        return False
    if state != 4:
        ctx.log(f"Cutter is not in IDLE, packml state: {state}")
        return False
    ctx.log("Cutter is in IDLE")
    return True


def read_live_value(ctx: WorkerContext):
    """Returns the value for the headweight page. Called when the page opens
    and each time Reload is pressed.
    """
    return ctx.get_mqtt().monitor("head_weight")


def supply_task(ctx: WorkerContext):
    global disk_position
    ctx.log(f"Preparing for supply to {ctx.selection.machine_id}")
    ctx.get_transport().run_command("supply", ctx.stop_event, timeout=300)
    ctx.log("Supply complete")
    disk_position= "A_side"


def transition_task(ctx: WorkerContext):
    """Get the disc ready for the next run, based on disk_position:
        No Disk -> supply (shouldn't happen, but handled)
        A_side  -> flip
        B_side  -> operator manually rejects the disc, then supply"""
    global disk_position
    transport = ctx.get_transport()
    machine = ctx.selection.machine_id
    ctx.log(f"Switch over: disc is currently {disk_position}")

    if disk_position == "No Disk":
        ctx.logger.warning(f"No disc recorded in {machine} at switch over - requesting supply")
        transport.run_command("supply", ctx.stop_event, timeout=300)
        disk_position = "A_side"

    elif disk_position == "A_side":
        ctx.log(f"Requesting flip for {machine}")
        transport.run_command("flip", ctx.stop_event, timeout=300)
        disk_position = "B_side"

    elif disk_position == "B_side":
        ctx.log("Disc has been used on both sides - manual reject required")
        ctx.ask_operator("Please manually remove and reject the disc currently in the cutter.\n\n"
                         "Press CONFIRM once it has been removed.")
        # Recorded straight away, so if the supply fails, RETRY supplies without asking again
        disk_position = "No Disk"
        ctx.log(f"Requesting supply for {machine}")
        transport.run_command("supply", ctx.stop_event, timeout=300)
        disk_position = "A_side"

    else:
        raise RuntimeError(f"Unknown disk position: {disk_position!r}")

    ctx.log(f"Switch over complete: disc is now {disk_position}")


def choose_after_transition(ctx: WorkerContext):
    """Where the switch over goes next. Set by choose_after_calibration, depending on
    which page leads into the switch over: dataCalcs -> validation, manual_adjust -> Calibration."""
    target = ctx.values.get("after_transition")
    if target is None:
        raise RuntimeError("No destination set for after the switch over")
    ctx.log(f"Switch over -> {target}")
    return target


def choose_after_diamond(ctx: WorkerContext):
    """ After the diamond in the cutter has been confirmed we check whether this is a calibration after a diamond has already been calibrated
    or if this is the first calibration for that cutter
    """
    global length_change

    first_calibration = False
    change_in_diamond_length = airtable.diamond_length_change(ctx.selection.machine_id)
    length_change = change_in_diamond_length
    ctx.log(f"Change in diamond length: {change_in_diamond_length}")
    if change_in_diamond_length is False:
        first_calibration = True
        ctx.log(f"Identified as first calibraiton on this cutter moving on to set default headweight")

    return "live_value" if first_calibration else "supply"



def run1_worker(ctx: WorkerContext):
    """ Run the camera calibration on the A side it takes in the diamond length change (none if this is the first calibration) """
    # The calibration sets its stop event when it finishes, so give it its own.
    # STOP PROGRAM is forwarded to it; the calibration finishing doesn't stop the app.
    global length_change

    new_drop_in = ctx.values.pop("manual_adj_new_drop_in", None)

        # Results from any earlier run no longer apply
    for key in ("side_a_csv", "mmum", "y_int", "manual_adj_req",
                "manual_ajd_instr", "analysis_error"):
        ctx.values.pop(key, None)
    cal_stop = threading.Event()

    def forward_stop():
        while not cal_stop.is_set():
            if ctx.stop_event.wait(0.2):
                cal_stop.set()

    threading.Thread(target=forward_stop, name="cal_stop_link", daemon=True).start()

    # length_change is False for a first calibration; treat that as no change
    change = length_change

    api, mqtt = ctx.get_api(), ctx.get_mqtt()

    if new_drop_in is not None:
        ctx.log(f"Using drop-in position {new_drop_in:.2f} from the manual adjustment")

    cal = CalibrationSide(ctx.logger, ctx.selection.log_folder, cal_stop, api, mqtt, change, frame_sink=ctx.frames.set, new_drop_in=new_drop_in)
    try:
        csv_path = cal.run_calibration(ctx.selection.machine_id)
    finally:
        cal_stop.set()   # also ends forward_stop
    ctx.values["side_a_csv"] = csv_path
    ctx.log(f"Side A samples saved to {csv_path}")

    # --- analysis: errors are logged, never re-run the calibration --- #
    if ctx.stop_event.is_set():
        ctx.log("Stopped before analysis")
        return
    if not csv_path:
        ctx.logger.warning("No samples recorded, skipping analysis")
        ctx.values["analysis_error"] = "No samples recorded"
        return

    try:
        ctx.log("Running analysis on side A data")
        la = LinearAnalysis(ctx.logger, ctx.selection.log_folder, csv_path)
        fit, adjustments = la.run_analysis()
        manual_req, shift = adjustments[0], adjustments[1]
        ctx.values["mmum"] = fit["gradient"]
        ctx.values["y_int"] = fit["intercept"]
        ctx.values["manual_adj_req"] = manual_req
        ctx.values["manual_ajd_instr"] = shift
        
        
        if ctx.values["manual_adj_req"]:
            shift = ctx.values["manual_ajd_instr"]
            if shift is None:
                ctx.logger.warning("Spring replacement required - the next run can't use "
                                   "this fit for its drop-in position")
            else:
                # The head shift moves the line by the same amount (check the sign)
                y_int_after = fit["intercept"] + float(shift)
                ctx.values["manual_adj_new_drop_in"] = fit["gradient"] * cutter.DROP_IN_AIM + y_int_after

        ctx.log(f"Analysis complete: gradient {fit['gradient']}, intercept {fit['intercept']}, "
                f"manual adjustment required: {adjustments[0]}")
    except Exception as e:
        ctx.logger.exception("Analysis failed - calibration data is saved and can be re-analysed")
        ctx.values["analysis_error"] = str(e)
    
def measure_and_set_headweight(ctx: WorkerContext) -> bool:
    """Find the default headweight. If it can be set at the default groove width, post and
    apply the new config and return True. Otherwise store the readings for the
    LoadCellShim page and return False."""
    api = ctx.get_api()
    mqtt = ctx.get_mqtt()

    # Results from any earlier attempt no longer apply
    for key in ("shim_required", "shim_headweight", "shim_piezo_pos"):
        ctx.values.pop(key, None)

    mmum = float(ctx.values["mmum"])
    y_int = float(ctx.values["y_int"])
    default_piezo_pos = round(((mmum * cutter.DEFAULT_GW) + y_int), 2)
    ctx.values["default_piezo_pos"] = default_piezo_pos
    config_list = find_default_headweight(api, mqtt, default_piezo_pos)

    if not config_list[0]:
        ctx.values["shim_required"] = True
        ctx.values["shim_headweight"] = config_list[1]
        ctx.values["shim_piezo_pos"] = config_list[2]
        ctx.logger.warning(f"Load cell shim required: headweight {config_list[1]}, "
                           f"piezo pos {config_list[2]} - config not changed")
        return False

    data = {
        "HEAD_DROP_AND_CALIBRATION_CFG_DEFAULT_WEIGHT": float(config_list[1]),
        "HEAD_DROP_AND_CALIBRATION_CFG_GRAMS_PER_MICRON": mmum,
        "HEAD_DROP_AND_CALIBRATION_CFG_MM_PER_GRAM": 1,
    }
    api.post_config(data)
    api.command("config_apply")
    ctx.logger.info(f"New calibration values set on cutter: {data}")
    return True
    
        #FUTURE DEVELOPMENT
        # default_headweight = config_list[1]
        # piezo_pos = config_list[2]
        # #using the piezo pos find the groove width
        # default_gw = round(((piezo_pos - y_int) / mmum), 2)
        # data = {
        #     "HEAD_DROP_AND_CALIBRATION_CFG_DEFAULT_WIDTH": float(default_gw),
        #     "HEAD_DROP_AND_CALIBRATION_CFG_DEFAULT_WEIGHT": float(default_headweight),
        #     "HEAD_DROP_AND_CALIBRATION_CFG_GRAMS_PER_MICRON": float(mmum),
        #     "HEAD_DROP_AND_CALIBRATION_CFG_MM_PER_GRAM": 1
        #     }
    
    # api.post_config(data)
    # api.command("conifg_apply")
    # ctx.logger.info(f"New calibration values set on cutter: {data}")

def get_image(ctx: WorkerContext):
    """Return the png image in the file of the linear regession with error margins"""
    image_folder = ctx.selection.log_folder

    if image_folder is None:
        return None

    pngs = list(image_folder.glob("*.png"))
    if not pngs:
        return None
    if len(pngs) > 1:
        ctx.logger.warning(f"Found {len(pngs)} PNG files in {image_folder}, using the newest")
    return max(pngs, key=lambda p: p.stat().st_mtime)

def choose_after_calibration(ctx: WorkerContext):
    """After the calibration: manual adjustment page if the analysis asked for one,
    otherwise straight on to finding the default headweight.
    Also sets where the switch over goes afterwards: back to Calibration after a
    manual adjustment, or on to validation after the headweight calculation."""
    if "analysis_error" in ctx.values:
        raise RuntimeError(f"Analysis failed: {ctx.values['analysis_error']}. "
                           "The calibration data has been saved - see the log.")
    
    if "mmum" not in ctx.values or "y_int" not in ctx.values:
        raise RuntimeError("Analysis did not produce a fit - see the log.")

    if ctx.values.get("manual_adj_req"):
        ctx.log("Manual adjustment required before setting the default headweight")
        ctx.values["after_transition"] = "Calibration"
        return "manual_adjust"
    ctx.values["after_transition"] = "validation"
    return "dataCalcs"


def manual_adjust_text(ctx: WorkerContext):
    instruction = ctx.values.get("manual_ajd_instr")
    if instruction == None:
        instructions = "Replace the spring: Spring has been overstretched and can no longer keep head in acceptable range"
    elif float(instruction) > 0:
        instructions = f"Please shift the head UPWARDS by approx. {instruction}"
    else:
        instructions = f"Please shift the head DOWNWARDS by approx. {abs(instruction)}"

    return (f"A manual adjustment is required before continuing:\n\n{instructions}\n\n"
            "Press Next when the adjustment is done.")

def validation_worker(ctx: WorkerContext):
    """Cut the validation grooves and check the measured widths, showing the camera feed."""
    api, mqtt = ctx.get_api(), ctx.get_mqtt()
    ctx.values.pop("validation_result", None)

    # Own stop event, with STOP PROGRAM forwarded to it
    cal_stop = threading.Event()

    def forward_stop():
        while not cal_stop.is_set():
            if ctx.stop_event.wait(0.2):
                cal_stop.set()

    threading.Thread(target=forward_stop, name="val_stop_link", daemon=True).start()

    val = ValidationSide(ctx.logger, ctx.selection.log_folder, cal_stop, api, mqtt,
                         ctx.values["mmum"], ctx.values["y_int"], frame_sink=ctx.frames.set)
    try:
        pass_fail = val.run_validation(ctx.selection.machine_id)
    finally:
        cal_stop.set()   # also ends forward_stop

    if pass_fail is None:
        raise RuntimeError("Validation was stopped before it finished")
    result = "PASS" if pass_fail else "FAIL"
    ctx.values["validation_result"] = result
    ctx.log(f"Result of validation: {result}")

def load_cell_shim_text(ctx: WorkerContext):
    headweight = ctx.values.get("shim_headweight")
    piezo_pos = float(ctx.values.get("shim_piezo_pos"))
    default_piezo_pos =  float(ctx.values.get("default_piezo_pos"))

    shim_dist = round(abs(piezo_pos - default_piezo_pos), cutter.ROUND_SHIM) + cutter.SHIM_OFFSET
    return ("The default headweight could not be set at the current loadcell height.\n\n"
            f"Please shim the loadcell upwards by {shim_dist} from its current position\n"
            "Please shim the load cell, then press OK to measure again.")

def finish_text(ctx: WorkerContext):
    result = ctx.values.get("validation_result")
    machine = ctx.selection.machine_id
    if result == "PASS":
        return (f"Calibration of cutter {machine} is complete.\n\n"
                f"Validation: PASS - all groove widths are within ±{ABS_ERROR_MARGIN} um.\n\n"
                "Press Finish to close.")
    if result == "FAIL":
        return (f"Calibration of cutter {machine} did not pass validation.\n\n"
                f"Validation: FAIL - one or more groove widths were more than ±{ABS_ERROR_MARGIN} um "
                "from target. See the log for details.\n\n"
                "Do not return this cutter to production. Press Finish to close.")
    return "Validation did not run, so the calibration result is unknown."


def run_gui(logger):
    app = WizardApp(logger)
    app.mainloop()


if __name__ == "__main__":
    run_gui(setup_logging())