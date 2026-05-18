from __future__ import annotations

import logging

from qxg_platform.config import PlatformConfig
from qxg_platform.detection import DetectionTracker
from qxg_platform.domain import create_camera_object
from qxg_platform.inputs import InputHandler
from qxg_platform.qxg_builder import QXGBuilder
from qxg_platform.relevance import RelevanceSelector
from qxg_platform.visualization import Visualizer

LOGGER = logging.getLogger(__name__)


class QXGPlatform:
    def __init__(self, config: PlatformConfig, input_handler: InputHandler):
        self.config = config
        self.input_handler = input_handler
        self.camera = create_camera_object(config.raw)
        analysis = config.section("analysis") | {"reasoning_mode": config.reasoning_mode}
        self.detector = DetectionTracker(config.section("detection"), config.reasoning_mode)
        self.builder = QXGBuilder(analysis, self.camera)
        self.relevance = RelevanceSelector(config.section("relevance"))
        self.visualizer = Visualizer(config.section("visualization"))
        self.camera_id = self.camera.tracking_id if self.camera else 0

    def run(self) -> None:
        try:
            LOGGER.info("Platform processing loop started")
            for frame_idx, (frame, world_info) in enumerate(self.input_handler.frames(), start=1):
                objects = self.detector.process_frame(frame, world_info)
                relations, all_objects = self.builder.build(objects, frame_idx)
                relevant = self.relevance.select(all_objects, relations, self.camera_id)
                key = self.visualizer.display(frame, all_objects, relevant, relations)
                if frame_idx % 30 == 0:
                    LOGGER.info(
                        "Processed frame=%s objects=%s relations=%s relevant=%s",
                        frame_idx,
                        len(all_objects),
                        len(relations),
                        len(relevant),
                    )
                if key == ord("q"):
                    LOGGER.info("Stop requested from visualization window")
                    break
        except Exception:
            LOGGER.exception("Platform processing loop failed")
            raise
        finally:
            self.input_handler.close()
            self.visualizer.close()
            LOGGER.info("Platform resources released")
