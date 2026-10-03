from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import cv2
import numpy as np

from .models import EventRecord


class EventStore(Protocol):
    def save_event(self, event: EventRecord) -> None: ...

    def save_system_event(self, event_type: str, payload: dict) -> None: ...

    def save_face(self, face_id: str, embedding: np.ndarray, name: str, date_of_birth: str, image_path: str | None) -> None: ...

    def load_faces(self) -> dict[str, list[float]]: ...


class MongoEventStore:
    def __init__(self, uri: str, database_name: str, logger):
        from pymongo import ASCENDING, MongoClient

        self.client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        self.client.admin.command("ping")
        self.collection = self.client[database_name]["events"]
        self.system_events = self.client[database_name]["system_events"]
        self.faces = self.client[database_name]["faces"]
        self.collection.create_index([("face_id", ASCENDING), ("timestamp", ASCENDING)])
        self.system_events.create_index([("event_type", ASCENDING), ("timestamp", ASCENDING)])
        self.logger = logger

    def save_event(self, event: EventRecord) -> None:
        self.collection.insert_one(event.as_document())

    def save_system_event(self, event_type: str, payload: dict) -> None:
        self.system_events.insert_one({
            "event_id": uuid4().hex,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc),
            "payload": payload,
        })

    def save_face(self, face_id: str, embedding: np.ndarray, name: str, date_of_birth: str, image_path: str | None) -> None:
        self.faces.update_one({"face_id": face_id}, {"$set": {"face_id": face_id, "embedding": embedding.tolist(), "name": name, "date_of_birth": date_of_birth, "image_path": image_path, "registered_at": datetime.now(timezone.utc)}}, upsert=True)

    def load_faces(self) -> dict[str, list[float]]:
        return {item["face_id"]: item for item in self.faces.find({}, {"_id": 0, "face_id": 1, "embedding": 1, "name": 1, "date_of_birth": 1, "image_path": 1, "registered_at": 1})}

    def close(self) -> None:
        self.client.close()

    def reconcile_open_events(self, image_root: Path, logger) -> int:
        latest_by_face = {}
        for item in self.collection.find({}, {"_id": 0}).sort("timestamp", 1):
            latest_by_face[item["face_id"]] = item
        closed = 0
        for item in latest_by_face.values():
            if item.get("event_type") != "entry":
                continue
            source = Path(item["image_path"])
            if not source.is_absolute():
                source = Path.cwd() / source
            crop = cv2.imread(str(source))
            if crop is None:
                logger.warning("open_entry_crop_missing face_id=%s image=%s", item["face_id"], source)
                continue
            timestamp = datetime.now(timezone.utc)
            event_id = uuid4().hex
            image_path = save_crop(image_root, crop, "exit", timestamp, event_id)
            exit_event = dict(item)
            exit_event.update({"event_id": event_id, "event_type": "exit", "timestamp": timestamp, "image_path": image_path})
            self.collection.insert_one(exit_event)
            logger.info("open_entry_reconciled face_id=%s exit=%s", item["face_id"], timestamp.isoformat())
            closed += 1
        return closed


class FileEventStore:
    """Deterministic offline store used for tests and camera runs without MongoDB."""

    def __init__(self):
        self.events: list[EventRecord] = []
        self.system_events: list[dict] = []
        self.faces: dict[str, list[float]] = {}

    def save_event(self, event: EventRecord) -> None:
        self.events.append(event)

    def save_system_event(self, event_type: str, payload: dict) -> None:
        self.system_events.append({"event_type": event_type, "payload": payload})

    def save_face(self, face_id: str, embedding: np.ndarray, name: str, date_of_birth: str, image_path: str | None) -> None:
        self.faces[face_id] = {"embedding": embedding.tolist(), "name": name, "date_of_birth": date_of_birth, "image_path": image_path, "registered_at": datetime.now(timezone.utc)}

    def load_faces(self) -> dict[str, list[float]]:
        return dict(self.faces)


def save_crop(root: Path, crop: np.ndarray, event_type: str, timestamp: datetime, event_id: str) -> str:
    date_folder = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")
    folder_name = {"entry": "entries", "exit": "exits", "registration": "registrations"}[event_type]
    folder = root / folder_name / date_folder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{timestamp.astimezone(timezone.utc).strftime('%H%M%S_%f')}_{event_id}.jpg"
    if crop.size == 0 or not cv2.imwrite(str(path), crop):
        raise IOError(f"Unable to save face crop to {path}")
    return str(path)
