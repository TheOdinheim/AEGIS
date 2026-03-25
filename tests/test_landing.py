"""Tests for the AEGIS landing page at /."""

from fastapi.testclient import TestClient


class TestLandingPage:
    """Test landing page serving and content."""

    def test_landing_returns_200(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_landing_serves_html(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_landing_no_auth_required(self):
        """Landing page must be publicly accessible."""
        from main import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Authentication required" not in resp.text

    def test_landing_contains_brand(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/")
        assert "AEGIS" in resp.text
        assert "The AI Immune System" in resp.text

    def test_landing_contains_architecture(self):
        from main import app
        client = TestClient(app)
        text = client.get("/").text
        assert "Seven Layers" in text
        assert "L1" in text
        assert "L7" in text
        assert "Barrier Layer" in text
        assert "Self-Healing" in text

    def test_landing_contains_validation_metrics(self):
        from main import app
        client = TestClient(app)
        text = client.get("/").text
        assert "3,139" in text
        assert "96.36%" in text
        assert "0%" in text

    def test_landing_contains_compliance(self):
        from main import app
        client = TestClient(app)
        text = client.get("/").text
        assert "NIST AI RMF" in text
        assert "EU AI Act" in text
        assert "SOC 2" in text

    def test_landing_contains_dashboard_link(self):
        from main import app
        client = TestClient(app)
        text = client.get("/").text
        assert "/dashboard" in text

    def test_landing_contains_contact(self):
        from main import app
        client = TestClient(app)
        text = client.get("/").text
        assert "odinheimllc@gmail.com" in text
        assert "Odin LLC" in text

    def test_landing_does_not_break_health(self):
        """Landing page route must not conflict with /health."""
        from main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code in (200, 503)

    def test_landing_does_not_break_dashboard(self):
        """Landing page route must not conflict with /dashboard."""
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
