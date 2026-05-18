from __future__ import annotations

import json

import numpy as np

from qxg_platform.inputs import CameraIntrinsics, DepthFrame
from qxg_platform.recording import FrameRecorder


def test_frame_recorder_saves_color_only_for_2d(tmp_path) -> None:
    recorder = FrameRecorder({"enabled": True, "output_dir": str(tmp_path)}, "2d")
    frame = np.full((8, 10, 3), 120, dtype=np.uint8)

    recorder.save(1, frame, None)

    session_dir = next(tmp_path.iterdir())
    assert (session_dir / "color" / "000001-color.jpg").exists()
    assert not (session_dir / "depth").exists()
    assert not (session_dir / "config.json").exists()


def test_frame_recorder_saves_depth_and_metadata_for_3d(tmp_path) -> None:
    recorder = FrameRecorder({"enabled": True, "output_dir": str(tmp_path)}, "3d")
    frame = np.full((8, 10, 3), 120, dtype=np.uint8)
    depth = np.full((8, 10), 1.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(width=10, height=8, fx=12.0, fy=13.0, ppx=5.0, ppy=4.0)

    recorder.save(1, frame, DepthFrame(depth, intrinsics))

    session_dir = next(tmp_path.iterdir())
    assert (session_dir / "color" / "000001-color.jpg").exists()
    assert (session_dir / "depth" / "000001-depth.png").exists()
    metadata = json.loads((session_dir / "config.json").read_text(encoding="utf-8"))
    assert metadata["depth_scale"] == 1000.0
    assert metadata["cam_intr"][0][0] == 12.0
