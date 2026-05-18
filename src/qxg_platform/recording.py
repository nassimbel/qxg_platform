from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class FrameRecorder:
    def __init__(self, config: dict[str, Any], reasoning_mode: str):
        self.enabled = bool(config.get("enabled", False))
        self.reasoning_mode = reasoning_mode
        self.depth_scale = float(config.get("depth_scale", 1000.0))
        self.image_quality = int(config.get("image_quality", 95))
        self.session_dir: Path | None = None
        self.color_dir: Path | None = None
        self.depth_dir: Path | None = None
        self._metadata_written = False
        if not self.enabled:
            return

        output_root = Path(str(config.get("output_dir", "recordings"))).expanduser()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = (output_root / f"qxg_recording_{timestamp}").resolve()
        self.color_dir = self.session_dir / "color"
        self.depth_dir = self.session_dir / "depth"
        self.color_dir.mkdir(parents=True, exist_ok=True)
        if self.reasoning_mode == "3d":
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Recording frames to %s", self.session_dir)

    def save(self, frame_idx: int, frame: np.ndarray, world_info: Any | None) -> None:
        if not self.enabled or self.color_dir is None:
            return

        stem = f"{frame_idx:06d}"
        color_path = self.color_dir / f"{stem}-color.jpg"
        cv2.imwrite(str(color_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

        depth_frame = self._depth_frame(world_info)
        if self.reasoning_mode == "3d" and depth_frame is not None and self.depth_dir is not None:
            depth_path = self.depth_dir / f"{stem}-depth.png"
            depth_map = np.asarray(depth_frame.get_data())
            if depth_map.dtype == np.uint16:
                depth_to_write = depth_map
            else:
                depth_to_write = np.clip(depth_map * self.depth_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_to_write)
            if not self._metadata_written:
                self._write_metadata(frame, depth_frame)

    def _write_metadata(self, frame: np.ndarray, depth_frame: Any) -> None:
        if self.session_dir is None:
            return
        intrinsics = self._intrinsics(depth_frame)
        height, width = frame.shape[:2]
        data: dict[str, Any] = {
            "im_w": width,
            "im_h": height,
            "depth_scale": self.depth_scale,
            "reasoning_mode": self.reasoning_mode,
        }
        if intrinsics is not None:
            data["cam_intr"] = [
                [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
                [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
                [0.0, 0.0, 1.0],
            ]
        (self.session_dir / "config.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        self._metadata_written = True

    def _depth_frame(self, world_info: Any | None) -> Any | None:
        if world_info is None:
            return None
        if hasattr(world_info, "get_depth_frame"):
            return world_info.get_depth_frame()
        if hasattr(world_info, "get_data"):
            return world_info
        return None

    def _intrinsics(self, depth_frame: Any) -> Any | None:
        if hasattr(depth_frame, "profile"):
            try:
                return depth_frame.profile.as_video_stream_profile().get_intrinsics()
            except AttributeError:
                return None
        if hasattr(depth_frame, "get_intrinsics"):
            return depth_frame.get_intrinsics()
        return None
