from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    track_id: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    embedding: np.ndarray
    observed_at: datetime


@dataclass
class ActiveTrack:
    track_id: int
    face_id: str
    last_seen: datetime
    last_bbox: tuple[int, int, int, int]
    last_crop: np.ndarray
    name: str
    date_of_birth: str
    entry_written: bool = False
    exit_written: bool = False


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    face_id: str
    name: str
    date_of_birth: str
    event_type: str
    timestamp: datetime
    image_path: str
    track_id: int
    recognized: bool
    confidence: Optional[float]

    def as_document(self) -> dict:
        return {
            "event_id": self.event_id,
            "face_id": self.face_id,
            "name": self.name,
            "date_of_birth": self.date_of_birth,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "track_id": self.track_id,
            "recognized": self.recognized,
            "confidence": self.confidence,
        }
