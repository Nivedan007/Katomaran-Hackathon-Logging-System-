from src.config import Settings


def test_video_file_source_is_not_rtsp(monkeypatch):
    monkeypatch.setenv("VIDEO_SOURCE", "/tmp/sample.mp4")
    settings = Settings()
    assert settings.capture_source == "/tmp/sample.mp4"
    assert settings.is_rtsp is False


def test_rtsp_source_is_detected(monkeypatch):
    monkeypatch.setenv("VIDEO_SOURCE", "rtsps://camera.example/stream")
    settings = Settings()
    assert settings.capture_source == "rtsps://camera.example/stream"
    assert settings.is_rtsp is True
