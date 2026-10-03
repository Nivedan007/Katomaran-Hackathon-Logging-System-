from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from src.event_manager import EventManager
from src.models import FaceObservation
from src.recognition import EmbeddingRecognizer
from src.storage import FileEventStore


def test_each_track_gets_exactly_one_entry_and_exit(tmp_path: Path):
    logger = __import__("logging").getLogger("test")
    store = FileEventStore()
    recognizer = EmbeddingRecognizer(0.78, logger)
    manager = EventManager(store, tmp_path / "logs", 2.0, logger, recognizer)
    crop = np.full((20, 20, 3), 128, dtype=np.uint8)
    first = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for seconds in (0, 0.5, 1.0):
        observed = first + timedelta(seconds=seconds)
        manager.process(FaceObservation(7, (0, 0, 20, 20), crop, recognizer.embed(crop), observed))
    manager.expire_missing(first + timedelta(seconds=2.1))
    manager.expire_missing(first + timedelta(seconds=5))

    assert [event.event_type for event in store.events] == ["entry", "exit"]
    assert len(list((tmp_path / "logs").rglob("*.jpg"))) == 2
    assert (tmp_path / "logs" / "entries" / "2026-01-01").exists()
    assert (tmp_path / "logs" / "exits" / "2026-01-01").exists()


def test_non_matching_face_is_ignored_when_a_face_is_registered(tmp_path: Path):
    logger = __import__("logging").getLogger("test-private")
    store = FileEventStore()
    recognizer = EmbeddingRecognizer(0.78, logger, store)
    registered = np.full((20, 20, 3), 128, dtype=np.uint8)
    other = np.zeros((20, 20, 3), dtype=np.uint8)
    recognizer.register("me", registered)
    manager = EventManager(store, tmp_path / "logs", 2.0, logger, recognizer)

    manager.process(FaceObservation(9, (0, 0, 20, 20), other, recognizer.embed(other), datetime.now(timezone.utc)))

    assert store.events == []
    assert store.system_events[-1]["event_type"] == "face_ignored"
