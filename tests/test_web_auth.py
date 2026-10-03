from web import create_app


def test_dashboard_requires_login():
    client = create_app().test_client()
    assert client.get("/").status_code == 302
    assert client.get("/login").status_code == 200


def test_valid_login_opens_dashboard(monkeypatch):
    monkeypatch.setenv("LOGIN_USERNAME", "nivedan")
    monkeypatch.setenv("LOGIN_PASSWORD", "secret")
    client = create_app().test_client()
    response = client.post("/login", data={"username": "nivedan", "password": "secret"})
    assert response.status_code == 302
    assert response.location.endswith("/")