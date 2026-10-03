from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    face_id: str
    confidence: float


class EmbeddingRecognizer:
    """Compact OpenCV embedding store with audit persistence."""

    def __init__(self, threshold: float, logger, store=None, default_name: str = "nivedan", default_date_of_birth: str = "03/02/2005"):
        self.threshold = threshold
        self.logger = logger
        self.store = store
        saved = store.load_faces() if store is not None else {}
        self.profiles = {}
        for face_id, value in saved.items():
            profile = value if isinstance(value, dict) else {"embedding": value}
            profile.setdefault("name", default_name)
            profile.setdefault("date_of_birth", default_date_of_birth)
            self.profiles[face_id] = profile
            if store is not None and ("name" not in value or "date_of_birth" not in value):
                store.save_face(face_id, np.asarray(profile["embedding"], dtype=np.float32), profile["name"], profile["date_of_birth"], profile.get("image_path"))
        self.embeddings: dict[str, np.ndarray] = {face_id: np.asarray(value["embedding"], dtype=np.float32) for face_id, value in self.profiles.items()}

    @property
    def has_registrations(self) -> bool:
        return bool(self.embeddings)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32)).astype(np.float32) / 255.0
        vector = resized.flatten()
        norm = np.linalg.norm(vector)
        embedding = vector / norm if norm else vector
        self.logger.info("embedding_generated dimensions=%s", embedding.size)
        self._audit("embedding_generated", {"dimensions": int(embedding.size)})
        return embedding

    def register(self, face_id: str, crop: np.ndarray, image_path: str | None = None, name: str = "nivedan", date_of_birth: str = "03/02/2005") -> np.ndarray:
        embedding = self.embed(crop)
        self.embeddings[face_id] = embedding
        self.profiles[face_id] = {"embedding": embedding, "name": name, "date_of_birth": date_of_birth, "image_path": image_path}
        if self.store is not None:
            self.store.save_face(face_id, embedding, name, date_of_birth, image_path)
        self.logger.info("face_registered face_id=%s", face_id)
        payload = {"face_id": face_id, "name": name, "date_of_birth": date_of_birth, "dimensions": int(embedding.size)}
        if image_path:
            payload["image_path"] = image_path
        self._audit("face_registered", payload)
        return embedding

    def profile(self, face_id: str) -> dict:
        profile = self.profiles.get(face_id, {})
        return {"name": profile.get("name", face_id), "date_of_birth": profile.get("date_of_birth", "")}

    def identify(self, embedding: np.ndarray) -> Match | None:
        if not self.embeddings:
            return None
        best_id, best_score = max(
            ((face_id, float(np.dot(known, embedding))) for face_id, known in self.embeddings.items()),
            key=lambda item: item[1],
        )
        if best_score >= self.threshold:
            self.logger.info("face_recognized face_id=%s confidence=%.4f", best_id, best_score)
            self._audit("face_recognized", {"face_id": best_id, "confidence": best_score})
            return Match(best_id, best_score)
        return None

    def _audit(self, event_type: str, payload: dict) -> None:
        if self.store is not None:
            self.store.save_system_event(event_type, payload)
