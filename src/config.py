from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    mongodb_database: str = field(default_factory=lambda: os.getenv("MONGODB_DATABASE", "face_logging"))
    video_source: str = field(default_factory=lambda: os.getenv("VIDEO_SOURCE", os.getenv("CAMERA_INDEX", "0")))
    absence_seconds: float = field(default_factory=lambda: float(os.getenv("ABSENCE_SECONDS", "2.0")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.78")))
    face_name: str = field(default_factory=lambda: os.getenv("FACE_NAME", "nivedan"))
    date_of_birth: str = field(default_factory=lambda: os.getenv("DATE_OF_BIRTH", "03/02/2005"))
    login_username: str = field(default_factory=lambda: os.getenv("LOGIN_USERNAME", "nivedan"))
    login_password: str = field(default_factory=lambda: os.getenv("LOGIN_PASSWORD", "change-me"))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "local-development-secret"))
    web_port: int = field(default_factory=lambda: int(os.getenv("WEB_PORT", "5050")))
    rtsp_reconnect_attempts: int = field(default_factory=lambda: int(os.getenv("RTSP_RECONNECT_ATTEMPTS", "3")))
    rtsp_reconnect_delay: float = field(default_factory=lambda: float(os.getenv("RTSP_RECONNECT_DELAY", "2.0")))
    log_root: Path = field(default_factory=lambda: Path(os.getenv("LOG_ROOT", "logs")))

    @property
    def event_log_path(self) -> Path:
        return self.log_root / "events.log"

    @property
    def capture_source(self) -> int | str:
        return int(self.video_source) if self.video_source.isdigit() else self.video_source

    @property
    def is_rtsp(self) -> bool:
        return self.video_source.lower().startswith(("rtsp://", "rtsps://"))
