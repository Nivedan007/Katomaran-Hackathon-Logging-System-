from datetime import datetime, timezone
from pathlib import Path
import webbrowser

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
import hmac

from src.config import Settings


def create_app() -> Flask:
    settings = Settings()
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.secret_key = settings.secret_key

    @app.before_request
    def require_login():
        if request.endpoint in {"login", "static"}:
            return None
        if session.get("authenticated"):
            return None
        if request.path.startswith(("/api/", "/media/")):
            return jsonify({"ok": False, "error": "authentication_required"}), 401
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if hmac.compare_digest(username, settings.login_username) and hmac.compare_digest(password, settings.login_password):
                session["authenticated"] = True
                return redirect(url_for("dashboard"))
            error = "Invalid username or password"
        return render_template("login.html", error=error)

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    def mongo_database():
        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return client, client[settings.mongodb_database]

    @app.get("/")
    def dashboard():
        return render_template("index.html", source=settings.video_source, source_kind=_source_kind(settings))

    @app.get("/api/overview")
    def overview():
        client = None
        try:
            client, database = mongo_database()
            events = database.events
            system_events = database.system_events
            latest = list(events.find({}, {"_id": 0}).sort("timestamp", -1).limit(12))
            audit = list(system_events.find({}, {"_id": 0}).sort("timestamp", -1).limit(12))
            identities = list(database.faces.find({}, {"_id": 0, "face_id": 1, "name": 1, "date_of_birth": 1, "image_path": 1, "registered_at": 1}))
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_events = list(events.find({"timestamp": {"$gte": datetime.fromisoformat(today).replace(tzinfo=timezone.utc)}}))
            return jsonify({
                "ok": True,
                "database": settings.mongodb_database,
                "stats": {
                    "today_total": len(today_events),
                    "today_entries": sum(event.get("event_type") == "entry" for event in today_events),
                    "today_exits": sum(event.get("event_type") == "exit" for event in today_events),
                    "known_faces": len(database.faces.distinct("face_id")) if "faces" in database.list_collection_names() else 0,
                },
                "events": _json_safe(latest),
                "audit": _json_safe(audit),
                "identities": _json_safe(identities),
            })
        except Exception as error:
            return jsonify({"ok": False, "error": str(error), "events": [], "audit": []}), 503
        finally:
            if client:
                client.close()

    @app.get("/media/<path:filename>")
    def media(filename: str):
        root = settings.log_root.resolve()
        requested = (Path.cwd() / filename).resolve()
        if root not in requested.parents or not requested.is_file():
            return ("Not found", 404)
        return send_file(requested)

    return app


def _source_kind(settings: Settings) -> str:
    if settings.is_rtsp:
        return "RTSP stream"
    if isinstance(settings.capture_source, int):
        return "Camera"
    return "Development video"


def _json_safe(items: list[dict]) -> list[dict]:
    for item in items:
        for key in ("timestamp", "registered_at"):
            timestamp = item.get(key)
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                item[key] = timestamp.astimezone(timezone.utc).isoformat()
    return items


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
