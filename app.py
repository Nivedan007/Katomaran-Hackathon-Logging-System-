from datetime import datetime, timezone
import time

import cv2

from src.config import Settings
from src.event_manager import EventManager
from src.logging_setup import configure_logging
from src.models import FaceObservation
from src.recognition import EmbeddingRecognizer
from src.storage import FileEventStore, MongoEventStore
from src.tracking import CentroidTracker, detect_faces


def main() -> None:
    settings = Settings()
    logger = configure_logging(settings.event_log_path)
    try:
        store = MongoEventStore(settings.mongodb_uri, settings.mongodb_database, logger)
        logger.info("database_connected database=%s", settings.mongodb_database)
    except Exception as error:
        logger.warning("database_unavailable fallback=file_store error=%s", error)
        store = FileEventStore()
    reconcile = getattr(store, "reconcile_open_events", None)
    if reconcile:
        reconcile(settings.log_root, logger)

    recognizer = EmbeddingRecognizer(settings.similarity_threshold, logger, store, settings.face_name, settings.date_of_birth)
    if not recognizer.has_registrations:
        logger.warning("no_face_registered starting_interactive_enrollment=true")
        from enroll_camera import enroll_from_source

        if not enroll_from_source(settings, store, logger):
            raise RuntimeError("Face enrollment cancelled. Restart app.py and press 's' when your face is visible.")
        recognizer = EmbeddingRecognizer(settings.similarity_threshold, logger, store, settings.face_name, settings.date_of_birth)
    manager = EventManager(store, settings.log_root, settings.absence_seconds, logger, recognizer)
    tracker = CentroidTracker(retention_seconds=settings.absence_seconds)
    camera = _open_source(settings, logger)
    if not camera.isOpened():
        camera.release()
        logger.error("video_source_unavailable source=%s", settings.video_source)
        close = getattr(store, "close", None)
        if close:
            close()
        raise RuntimeError(
            f"Unable to open video source '{settings.video_source}'. "
            "On macOS, allow camera access for VS Code or your terminal in "
            "System Settings > Privacy & Security > Camera, then restart it. "
            "For a video file, set VIDEO_SOURCE=/absolute/path/to/video.mp4 in .env."
        )

    logger.info("video_source_started source_type=%s source=%s", _source_type(settings), settings.video_source)
    blank_frames = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                if isinstance(settings.capture_source, int):
                    logger.error("camera_frame_read_failed source=%s", settings.video_source)
                    raise RuntimeError(
                        "Camera opened but returned no frames. Check macOS Camera permission, "
                        "close other apps using the camera, or set VIDEO_SOURCE to a video file."
                    )
                if not settings.is_rtsp:
                    logger.info("video_source_ended source=%s", settings.video_source)
                    break
                logger.warning("rtsp_frame_read_failed reconnecting=true")
                camera.release()
                camera = _reconnect_rtsp(settings, logger)
                if camera is None:
                    break
                continue
            if isinstance(settings.capture_source, int) and float(frame.mean()) < 1.0:
                blank_frames += 1
                frame[:] = (35, 42, 45)
                cv2.putText(frame, "CAMERA SIGNAL UNAVAILABLE", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3)
                cv2.putText(frame, "Enable Camera access for Terminal or VS Code", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)
                cv2.putText(frame, "Close FaceTime / Zoom / Photo Booth, then restart this app", (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)
                cv2.putText(frame, "Press q to quit", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 220, 150), 2)
                cv2.imshow("Face Logging System - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if blank_frames >= 30:
                    raise RuntimeError("Camera returned black frames. Enable camera access for Terminal/VS Code, close other camera apps, and check the physical camera shutter.")
                continue
            blank_frames = 0
            now = datetime.now(timezone.utc)
            detections = tracker.update(detect_faces(frame), now)
            for detection in detections:
                x, y, width, height = detection.bbox
                crop = frame[y:y + height, x:x + width]
                if crop.size == 0:
                    continue
                embedding = recognizer.embed(crop)
                manager.process(FaceObservation(detection.track_id, detection.bbox, crop, embedding, now))
                cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 210, 120), 2)
                cv2.putText(frame, f"ID {detection.track_id}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 210, 120), 2)
            manager.expire_missing(now)
            cv2.imshow("Face Logging System - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.close()
        camera.release()
        cv2.destroyAllWindows()
        close = getattr(store, "close", None)
        if close:
            close()
        logger.info("camera_stopped")


def _source_type(settings: Settings) -> str:
    if settings.is_rtsp:
        return "rtsp"
    if isinstance(settings.capture_source, int):
        return "camera"
    return "video_file"


def _open_source(settings: Settings, logger):
    backend = cv2.CAP_AVFOUNDATION if isinstance(settings.capture_source, int) else cv2.CAP_ANY
    camera = cv2.VideoCapture(settings.capture_source, backend)
    if camera.isOpened():
        return camera
    logger.error("video_source_open_failed source_type=%s source=%s", _source_type(settings), settings.video_source)
    return camera


def _reconnect_rtsp(settings: Settings, logger):
    for attempt in range(1, settings.rtsp_reconnect_attempts + 1):
        time.sleep(settings.rtsp_reconnect_delay)
        camera = _open_source(settings, logger)
        if camera.isOpened():
            logger.info("rtsp_reconnected attempt=%s", attempt)
            return camera
        camera.release()
        logger.warning("rtsp_reconnect_failed attempt=%s", attempt)
    logger.error("rtsp_unavailable attempts=%s", settings.rtsp_reconnect_attempts)
    return None


if __name__ == "__main__":
    main()
