from __future__ import annotations

import argparse

from qxg_platform.config import load_config
from qxg_platform.inputs import RealtimeInput, RecordingInput, VideoFileInput, WebcamInput
from qxg_platform.logging_utils import configure_logging
from qxg_platform.platform import QXGPlatform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QXG Platform")
    parser.add_argument("--config", default="configs/video.yaml")
    parser.add_argument("--mode", choices=["local", "remote"], default="local")
    parser.add_argument(
        "--input", choices=["recording", "video", "camera", "realsense"], default="recording"
    )
    parser.add_argument("--source", default="D:/nassim/qxg_artifacts/recordings/clip1")
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    return parser


def main() -> None:
    log_file = configure_logging()
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.mode == "remote":
        raise NotImplementedError(
            "Remote client mode is intentionally separated from local inference."
        )
    if args.input == "realsense":
        input_handler = RealtimeInput(config.section("realsense"), config.reasoning_mode)
    elif args.input == "camera":
        input_handler = WebcamInput(int(args.source))
    elif args.input == "video":
        input_handler = VideoFileInput(args.source)
    else:
        input_handler = RecordingInput(args.source, config.reasoning_mode)
    try:
        QXGPlatform(config, input_handler).run()
    except Exception as exc:
        raise SystemExit(f"{exc}\nFull log: {log_file}") from exc


if __name__ == "__main__":
    main()
