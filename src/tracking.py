from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import cv2


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    track_id: int


class CentroidTracker:
    def __init__(self, max_distance: int = 80, retention_seconds: float = 2.0):
        self.max_distance = max_distance
        self.retention = timedelta(seconds=retention_seconds)
        self.next_id = 1
        self.centroids: dict[int, tuple[int, int]] = {}
        self.last_seen: dict[int, datetime] = {}

    def update(self, boxes: list[tuple[int, int, int, int]], observed_at: datetime | None = None) -> list[Detection]:
        observed_at = observed_at or datetime.now()
        result: list[Detection] = []
        unmatched = {track_id for track_id, last_seen in self.last_seen.items() if observed_at - last_seen <= self.retention}
        for box in boxes:
            x, y, width, height = box
            center = (x + width // 2, y + height // 2)
            candidate = min(unmatched, key=lambda item: _distance(center, self.centroids[item]), default=None)
            if candidate is None or _distance(center, self.centroids[candidate]) > self.max_distance:
                candidate = self.next_id
                self.next_id += 1
            else:
                unmatched.remove(candidate)
            self.centroids[candidate] = center
            self.last_seen[candidate] = observed_at
            result.append(Detection(box, candidate))
        expired = [track_id for track_id, last_seen in self.last_seen.items() if observed_at - last_seen > self.retention]
        for track_id in expired:
            del self.centroids[track_id]
            del self.last_seen[track_id]
        return result


def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def detect_faces(frame):
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "This project requires OpenCV 4.x. Reinstall dependencies with "
            "'pip install -r requirements.txt' inside the project virtual environment."
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    return [tuple(map(int, box)) for box in detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))]
