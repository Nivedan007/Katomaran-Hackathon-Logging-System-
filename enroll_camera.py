from datetime import datetime, timezone

import cv2

from src.config import Settings
from src.logging_setup import configure_logging
from src.recognition import EmbeddingRecognizer
from src.storage import MongoEventStore, save_crop
from src.tracking import detect_faces


def enroll_from_source(settings, store, logger) -> bool:
    camera = cv2.VideoCapture(settings.capture_source)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open source '{settings.video_source}' for enrollment")

    print("Look at the camera. Press 's' to save your face, or 'q' to cancel.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Unable to read an enrollment frame")
            faces = detect_faces(frame)
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 210, 120), 2)
            cv2.imshow("Face enrollment - press s to save", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("s"):
                if not faces:
                    print("No face detected; move closer and try again.")
                    continue
                x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
                crop = frame[y:y + height, x:x + width]
                timestamp = datetime.now(timezone.utc)
                image_path = save_crop(settings.log_root, crop, "registration", timestamp, "me")
                EmbeddingRecognizer(settings.similarity_threshold, logger, store).register("me", crop, image_path, settings.face_name, settings.date_of_birth)
                print(f"Face registered successfully. Image saved to {image_path}")
                return True
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    settings = Settings()
    logger = configure_logging(settings.event_log_path)
    store = MongoEventStore(settings.mongodb_uri, settings.mongodb_database, logger)
    try:
        enroll_from_source(settings, store, logger)
    finally:
        store.close()


if __name__ == "__main__":
    main()
