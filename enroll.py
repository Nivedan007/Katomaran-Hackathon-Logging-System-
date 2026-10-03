import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.config import Settings
from src.logging_setup import configure_logging
from src.recognition import EmbeddingRecognizer
from src.storage import MongoEventStore, save_crop
from src.tracking import detect_faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Register one face for the private logger")
    parser.add_argument("image", type=Path, help="Image containing your face")
    parser.add_argument("--face-id", default="me", help="Stable identity label stored in MongoDB")
    args = parser.parse_args()
    settings = Settings()
    logger = configure_logging(settings.event_log_path)
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Unable to read image: {args.image}")
    faces = detect_faces(image)
    if not faces:
        raise SystemExit("No face found in the enrollment image")
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    crop = image[y:y + height, x:x + width]
    store = MongoEventStore(settings.mongodb_uri, settings.mongodb_database, logger)
    try:
        image_path = save_crop(settings.log_root, crop, "registration", datetime.now(timezone.utc), args.face_id)
        EmbeddingRecognizer(settings.similarity_threshold, logger, store).register(args.face_id, crop, image_path, settings.face_name, settings.date_of_birth)
        print(f"Registered {args.face_id}; image saved to {image_path}")
    finally:
        store.close()


if __name__ == "__main__":
    main()