from __future__ import annotations

import copy
import logging
import queue
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk

import yaml

from qxg_platform.config import PlatformConfig, load_config
from qxg_platform.inputs import RealtimeInput, RecordingInput, VideoFileInput, WebcamInput
from qxg_platform.logging_utils import configure_logging
from qxg_platform.platform import QXGPlatform

DEFAULT_CONFIG = Path("configs/video.yaml")
DEFAULT_PROFILES = Path("configs/model_profiles.yaml")
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_DETECTION_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "cat",
    "dog",
    "backpack",
    "umbrella",
    "handbag",
    "bottle",
    "cup",
    "chair",
    "couch",
    "potted plant",
    "laptop",
    "cell phone",
]
LOGGER = logging.getLogger(__name__)


class LauncherApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("QXG Platform Launcher")
        self.root.geometry("920x720")
        self.root.minsize(840, 660)
        self.log_file = configure_logging()
        self.events: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.profiles = load_profiles(DEFAULT_PROFILES)

        self.input_type = StringVar(value="video")
        self.source = StringVar(value="")
        self.config_path = StringVar(value=str(DEFAULT_CONFIG))
        self.profile_key = StringVar(value=self.first_profile_for("video"))
        self.model_path = StringVar(value="")
        self.reasoning_mode = StringVar(value="2d")
        self.enable_relevance = BooleanVar(value=False)
        self.save_recording = BooleanVar(value=False)
        self.recording_dir = StringVar(value=str(DEFAULT_RECORDINGS_DIR))
        self.class_vars = {
            class_name: BooleanVar(value=False) for class_name in DEFAULT_DETECTION_CLASSES
        }
        self.status = StringVar(
            value=f"Choose a source, then start the platform. Logs: {self.log_file}"
        )

        self._build()
        self._sync_profile_to_form()
        self._poll_events()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text="Input Source").grid(row=0, column=0, sticky="w", pady=(0, 6))
        input_box = ttk.Combobox(
            outer,
            textvariable=self.input_type,
            values=["video", "camera", "realsense", "recording"],
            state="readonly",
        )
        input_box.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        input_box.bind("<<ComboboxSelected>>", lambda _event: self._on_input_change())

        ttk.Label(outer, text="Source").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(outer, textvariable=self.source).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(outer, text="Browse", command=self._browse_source).grid(
            row=1, column=2, padx=(8, 0), pady=6
        )

        ttk.Label(outer, text="Model Profile").grid(row=2, column=0, sticky="w", pady=6)
        self.profile_box = ttk.Combobox(
            outer,
            textvariable=self.profile_key,
            values=list(self.profiles.keys()),
            state="readonly",
        )
        self.profile_box.grid(row=2, column=1, sticky="ew", pady=6)
        self.profile_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_profile_to_form())

        ttk.Label(outer, text="Model Weights").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(outer, textvariable=self.model_path).grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Button(outer, text="Browse", command=self._browse_model).grid(
            row=3, column=2, padx=(8, 0), pady=6
        )

        ttk.Label(outer, text="Reasoning").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(
            outer,
            textvariable=self.reasoning_mode,
            values=["2d", "3d"],
            state="readonly",
        ).grid(row=4, column=1, sticky="ew", pady=6)

        ttk.Label(outer, text="Config").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(outer, textvariable=self.config_path).grid(row=5, column=1, sticky="ew", pady=6)
        ttk.Button(outer, text="Browse", command=self._browse_config).grid(
            row=5, column=2, padx=(8, 0), pady=6
        )

        ttk.Checkbutton(
            outer,
            text="Enable relevance filtering",
            variable=self.enable_relevance,
        ).grid(row=6, column=1, sticky="w", pady=8)

        recording = ttk.LabelFrame(outer, text="Realtime Recording")
        recording.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        recording.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            recording,
            text="Save frames while the platform runs",
            variable=self.save_recording,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        ttk.Entry(recording, textvariable=self.recording_dir).grid(
            row=0, column=1, sticky="ew", padx=(8, 0), pady=8
        )
        ttk.Button(recording, text="Browse", command=self._browse_recording_dir).grid(
            row=0, column=2, sticky="e", padx=8, pady=8
        )

        objects = ttk.LabelFrame(outer, text="Objects To Detect")
        objects.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for index, (class_name, variable) in enumerate(self.class_vars.items()):
            ttk.Checkbutton(objects, text=class_name, variable=variable).grid(
                row=index // 4,
                column=index % 4,
                sticky="w",
                padx=10,
                pady=3,
            )

        buttons = ttk.Frame(outer)
        buttons.grid(row=9, column=1, sticky="ew", pady=(18, 8))
        ttk.Button(buttons, text="Start Platform", command=self._start).pack(side="left")
        ttk.Button(buttons, text="Quit", command=self.root.destroy).pack(side="left", padx=8)

        info = ttk.LabelFrame(outer, text="Visualization")
        info.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(18, 0))
        info.columnconfigure(0, weight=1)
        ttk.Label(
            info,
            text=(
                "The runtime opens an OpenCV dashboard with camera view, BEV map, "
                "relationship graph, relation table, and object metrics. Press q "
                "inside that window to stop."
            ),
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        ttk.Label(outer, textvariable=self.status).grid(
            row=11, column=0, columnspan=3, sticky="ew", pady=(16, 0)
        )

    def _browse_source(self) -> None:
        input_type = self.input_type.get()
        if input_type == "video":
            value = filedialog.askopenfilename(
                title="Choose video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            )
        elif input_type == "recording":
            value = filedialog.askdirectory(title="Choose recording directory")
        else:
            value = ""
            self.source.set("0" if input_type == "camera" else "")
        if value:
            self.source.set(value)

    def _browse_model(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose model weights",
            filetypes=[("Model weights", "*.pt *.pth"), ("All files", "*.*")],
        )
        if value:
            self.model_path.set(value)

    def _browse_config(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose config",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if value:
            self.config_path.set(value)

    def _browse_recording_dir(self) -> None:
        value = filedialog.askdirectory(title="Choose recording output directory")
        if value:
            self.recording_dir.set(value)

    def _on_input_change(self) -> None:
        input_type = self.input_type.get()
        self.profile_key.set(self.first_profile_for(input_type))
        if input_type == "camera":
            self.source.set("0")
        elif input_type == "realsense":
            self.source.set("")
        self._sync_profile_to_form()

    def first_profile_for(self, input_type: str) -> str:
        for key, profile in self.profiles.items():
            if input_type in profile.get("input_types", []):
                return key
        return next(iter(self.profiles))

    def _sync_profile_to_form(self) -> None:
        profile = self.profiles[self.profile_key.get()]
        self.model_path.set(str(profile.get("model_weights", "")))
        self.reasoning_mode.set(str(profile.get("reasoning_mode", "2d")))
        self._sync_class_checkboxes(profile.get("classes", []))
        label = profile.get("label", self.profile_key.get())
        self.status.set(f"Selected model profile: {label}")

    def _sync_class_checkboxes(self, selected_classes: list[str]) -> None:
        selected = set(selected_classes or DEFAULT_DETECTION_CLASSES)
        for class_name, variable in self.class_vars.items():
            variable.set(class_name in selected)

    def _selected_classes(self) -> list[str]:
        return [
            class_name
            for class_name, variable in self.class_vars.items()
            if variable.get()
        ]

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("QXG Platform", "The platform is already running.")
            return
        try:
            config = self._build_config()
            input_handler = self._build_input_handler(config)
        except Exception as exc:
            LOGGER.exception("Failed to prepare platform from GUI selections")
            messagebox.showerror("Cannot start", f"{exc}\n\nFull log:\n{self.log_file}")
            return
        self.worker = threading.Thread(
            target=self._run_platform,
            args=(config, input_handler),
            daemon=True,
        )
        self.worker.start()
        self.status.set("Platform is running. Press q in the visualization window to stop.")
        LOGGER.info(
            "Started platform from GUI input_type=%s source=%s profile=%s model=%s",
            self.input_type.get(),
            self.source.get(),
            self.profile_key.get(),
            self.model_path.get(),
        )

    def _build_config(self) -> PlatformConfig:
        loaded = load_config(self.config_path.get())
        raw = copy.deepcopy(loaded.raw)
        profile = self.profiles[self.profile_key.get()]
        reasoning_mode = self.reasoning_mode.get()
        if reasoning_mode == "3d" and self.input_type.get() in {"camera", "video"}:
            LOGGER.info(
                "3D requested for input_type=%s; using monocular depth estimation",
                self.input_type.get(),
            )
            messagebox.showinfo(
                "Estimated depth",
                "QXG will use monocular depth estimation for this source. "
                "It is useful for visualization and relative 3D reasoning, but RealSense "
                "depth is more accurate for metric distances.",
            )
        raw.setdefault("runtime", {})["reasoning_mode"] = reasoning_mode
        raw.setdefault("detection", {})["model_weights"] = self.model_path.get()
        raw["detection"]["confidence_threshold"] = float(
            profile.get("confidence_threshold", raw["detection"].get("confidence_threshold", 0.5))
        )
        raw["detection"]["image_size"] = int(
            profile.get("image_size", raw["detection"].get("image_size", 960))
        )
        selected_classes = self._selected_classes()
        if not selected_classes:
            raise ValueError("Select at least one object to detect.")
        raw["detection"]["classes"] = selected_classes
        raw.setdefault("relevance", {})["enabled"] = bool(self.enable_relevance.get())
        raw.setdefault("visualization", {})["enabled"] = True
        raw["recording"] = {
            "enabled": bool(self.save_recording.get()),
            "output_dir": self.recording_dir.get() or str(DEFAULT_RECORDINGS_DIR),
        }
        return PlatformConfig(raw=raw, source_path=loaded.source_path)

    def _build_input_handler(self, config: PlatformConfig):
        input_type = self.input_type.get()
        source = self.source.get().strip()
        if input_type == "video":
            if not source:
                raise ValueError("Choose a video file.")
            return VideoFileInput(
                source, config.reasoning_mode, config.section("depth_estimation")
            )
        if input_type == "recording":
            if not source:
                raise ValueError("Choose a recording directory.")
            return RecordingInput(source, config.reasoning_mode)
        if input_type == "camera":
            return WebcamInput(
                int(source or "0"),
                config.reasoning_mode,
                config.section("depth_estimation"),
            )
        return RealtimeInput(config.section("realsense"), config.reasoning_mode)

    def _run_platform(self, config: PlatformConfig, input_handler) -> None:
        try:
            QXGPlatform(config, input_handler).run()
            self.events.put("Platform stopped.")
            LOGGER.info("Platform stopped normally")
        except Exception:
            LOGGER.exception("Platform runtime failed")
            self.events.put(traceback.format_exc())

    def _poll_events(self) -> None:
        try:
            message = self.events.get_nowait()
        except queue.Empty:
            self.root.after(250, self._poll_events)
            return
        self.status.set(message.splitlines()[-1] if message else "Platform stopped.")
        if "Traceback" in message:
            messagebox.showerror("Runtime error", f"{message}\n\nFull log:\n{self.log_file}")
        self.root.after(250, self._poll_events)


def load_profiles(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    profiles = raw.get("profiles", {})
    if not profiles:
        raise ValueError(f"No model profiles found in {path}")
    return profiles


def main() -> None:
    configure_logging()
    root = Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
