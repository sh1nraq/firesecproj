import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from pathlib import Path
from typing import Tuple, Optional


class Detector:
    def __init__(
        self,
        model_path: Path,
        target_height: int = 640,
        iou_threshold: float = 0.2,
        min_confidence: float = 0.5,
        smoke_confidence: float = 0.75
        ):

        try:
            self.model = YOLO(str(model_path))
            self.target_height = target_height
            self.iou_threshold = iou_threshold
            self.min_confidence = min_confidence
            self.smoke_confidence = smoke_confidence
            self.names = self.model.model.names
            self.colors = {
                "fire": (0, 0, 255),
                "smoke": (128, 128, 128)
            }
        except Exception as e:
            raise

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = int(self.target_height * aspect_ratio)
        return cv2.resize(frame, (new_width, self.target_height))

    def draw_detection(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        class_name: str,
        confidence: float
    ) -> None:
        x1, y1, x2, y2 = box
        color = self.colors.get(class_name.lower(), (0, 255, 0))

        text = f"{class_name}: {confidence:.2f}"

        label_height = 30
        if y1 < label_height:
            text_y = y2 + label_height
            rect_y = y2
        else:
            text_y = y1 - 5
            rect_y = y1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        corner_length = 20
        thickness = 2

        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)

        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)

        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)

        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)

        cvzone.putTextRect(
            frame,
            text,
            (x1, text_y),
            scale=1.5,
            thickness=2,
            colorR=color,
            colorT=(255, 255, 255),  # White text
            font=cv2.FONT_HERSHEY_SIMPLEX,
            offset=5,
            border=2,
            colorB=(0, 0, 0),  # Black border
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        try:
            frame = self.resize_frame(frame)
            results = self.model(
                frame, iou=self.iou_threshold, conf=self.min_confidence)
            detection = None

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                sort_idx = np.argsort(-confidences)  # Descending order
                boxes = boxes[sort_idx]
                class_ids = class_ids[sort_idx]
                confidences = confidences[sort_idx]

                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    class_name = self.names[class_id]

                    if detection is None:
                        if "fire" == class_name.lower() and confidence >= self.min_confidence:
                            detection = "Fire"
                        elif "smoke" == class_name.lower() and confidence >= self.smoke_confidence:
                            detection = "Smoke"

                    self.draw_detection(frame, box, class_name, confidence)

            self._add_frame_info(frame, detection)

            return frame, detection

        except Exception as e:
            return frame, None