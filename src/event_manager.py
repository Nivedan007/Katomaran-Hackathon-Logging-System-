from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .models import ActiveTrack, EventRecord, FaceObservation
from .storage import EventStore, save_crop


class EventManager:
    """Owns the exactly-once entry/exit state transition for every track."""

    def __init__(self, store: EventStore, image_root, absence_seconds: float, logger, recognizer):
        self.store = store
        self.image_root = image_root
        self.absence_after = timedelta(seconds=absence_seconds)
        self.logger = logger
        self.recognizer = recognizer
        self.active: dict[int, ActiveTrack] = {}

    def process(self, observation: FaceObservation) -> None:
        now = observation.observed_at
        track = self.active.get(observation.track_id)
        if track is None:
            match = self.recognizer.identify(observation.embedding)
            if self.recognizer.has_registrations and match is None:
                self.logger.info("face_ignored reason=not_registered track_id=%s", observation.track_id)
                self.store.save_system_event("face_ignored", {"track_id": observation.track_id, "reason": "not_registered"})
                return
            face_id = match.face_id if match else f"unknown-{observation.track_id}"
            profile = self.recognizer.profile(face_id)
            track = ActiveTrack(observation.track_id, face_id, now, observation.bbox, observation.crop, profile["name"], profile["date_of_birth"])
            self.active[observation.track_id] = track
            self._write_event("entry", observation, track, match.confidence if match else None)
        else:
            track.last_seen = now
            track.last_bbox = observation.bbox
            track.last_crop = observation.crop
            self.logger.info("face_tracking track_id=%s face_id=%s", track.track_id, track.face_id)
            self.store.save_system_event("face_tracking", {
                "track_id": track.track_id,
                "face_id": track.face_id,
                "bbox": observation.bbox,
                "timestamp": now,
            })

    def expire_missing(self, now: datetime) -> None:
        expired = [track for track in self.active.values() if now - track.last_seen >= self.absence_after]
        for track in expired:
            observation = FaceObservation(track.track_id, track.last_bbox, track.last_crop, _empty_embedding(), now)
            self._write_event("exit", observation, track, None)
            del self.active[track.track_id]

    def close(self, now: datetime | None = None) -> None:
        self.expire_missing(now or datetime.now(timezone.utc) + self.absence_after)

    def _write_event(self, event_type: str, observation: FaceObservation, track: ActiveTrack, confidence: float | None) -> None:
        if event_type == "entry" and track.entry_written:
            return
        if event_type == "exit" and track.exit_written:
            return
        timestamp = observation.observed_at.astimezone(timezone.utc)
        event_id = uuid4().hex
        image_path = save_crop(self.image_root, observation.crop, event_type, timestamp, event_id)
        event = EventRecord(event_id, track.face_id, track.name, track.date_of_birth, event_type, timestamp, image_path, track.track_id, confidence is not None, confidence)
        self.store.save_event(event)
        self.logger.info("face_%s face_id=%s track_id=%s image=%s", event_type, track.face_id, track.track_id, image_path)
        if event_type == "entry":
            track.entry_written = True
        else:
            track.exit_written = True


def _empty_embedding():
    import numpy as np
    return np.zeros(1, dtype=np.float32)
