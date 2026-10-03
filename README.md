# Face Logging System

A Python + OpenCV + MongoDB camera logger. It creates exactly one entry event when a face track appears and exactly one exit event after the track has been absent for the configured timeout.

## Features

- Face detection with OpenCV Haar cascades.
- Centroid tracking with stable per-session track IDs.
- Optional in-process embedding recognition using normalized grayscale embeddings.
- MongoDB metadata in the `events` collection.
- MongoDB audit metadata in the `system_events` collection for recognition, tracking, embedding generation, and face registration.
- Cropped images under `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`.
- Mandatory `logs/events.log` covering entry, recognition, tracking, exit, embedding generation, and registration.
- Automatic file-store fallback for camera testing when MongoDB is unavailable.
- Configurable input modes: local camera, development video file, and RTSP/RTSPS stream.
- Automatic RTSP reconnect attempts for short network interruptions.
- Private face mode: only enrolled faces generate movement events; other faces are ignored and audited.
- Face-only crops: saved images contain the detected face bounding box, never the full camera frame.

## Setup

1. Install Python 3.11+ and MongoDB, then create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

2. Start MongoDB locally, or set `MONGODB_URI` in `.env`.
3. Grant camera access to the application launching Python. On macOS, open **System Settings > Privacy & Security > Camera**, enable access for Terminal or VS Code, then fully restart that application.

The dashboard uses a login page. Set `LOGIN_USERNAME`, `LOGIN_PASSWORD`, and `SECRET_KEY` in `.env` before starting the web server. The example defaults are suitable only for local development.

Start the dashboard and open it in your external system browser automatically:

```bash
python web.py
```

You can also open the login page manually in Safari, Chrome, or Firefox at `http://127.0.0.1:5050/login`. Use `/logout` to clear an existing session. `WEB_PORT` can be changed if that port is already in use.

4. Register your face before starting the logger. Use a clear image containing your face:

```bash
python enroll.py /absolute/path/to/your-face.jpg --face-id me
```

The embedding is stored in MongoDB and the enrollment crop is saved under `logs/registrations/YYYY-MM-DD/`. The logger will reject non-matching faces.

Alternatively, enroll directly from the camera:

```bash
python enroll_camera.py
```

Look at the camera and press `s` to save your face. Press `q` to cancel.

5. Run the camera logger:

```bash
python app.py
```

Press `q` to stop. The first run creates the `logs` directory. For a different camera, set `VIDEO_SOURCE=1` (or another camera index).

For development video input, set `VIDEO_SOURCE` to an absolute `.mp4` path:

```env
VIDEO_SOURCE=/Users/you/Videos/sample.mp4
```

For an interview RTSP stream, set the URL in `.env`:

```env
VIDEO_SOURCE=rtsp://username:password@camera-host:554/stream-path
```

The same detection, tracking, MongoDB, image, and audit-log pipeline is used for all input modes. RTSP reconnect behavior is controlled by `RTSP_RECONNECT_ATTEMPTS` and `RTSP_RECONNECT_DELAY`.

## Verify the exactly-once guarantee

```bash
pytest -q
```

MongoDB event documents include `event_id`, `face_id`, `name`, `date_of_birth`, `event_type`, `timestamp`, `image_path`, `track_id`, `recognized`, and `confidence`. The dashboard shows the person's name, date of birth, event type, and exact entry/exit timestamp. Registered embeddings are stored in the `faces` collection. System-event documents include `event_id`, `event_type`, `timestamp`, and a structured `payload`. The same critical events are also written to `logs/events.log`.
