"""
Tests de integración para la API REST.
Requiere que el modelo esté entrenado: python main.py

Ejecutar: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_STUDENT = {
    "backend_skill": 4,
    "frontend_skill": 2,
    "database_skill": 3,
    "testing_skill": 3,
    "documentation_skill": 2,
    "git_github_skill": 4,
    "agile_methodologies_skill": 3,
    "teamwork_skill": 4,
    "communication_skill": 3,
    "leadership_skill": 2,
    "organization_skill": 3,
    "projects_completed": 5,
    "available_hours_per_week": 20,
    "last_role": "Backend",
    "last_role_performance": 4,
    "peer_evaluation": 4,
}

VALID_TASK = {
    "task_type": "Backend",
    "task_complexity": "Medium",
    "priority": "High",
    "estimated_hours": 15,
}


class TestHealthEndpoints:
    def test_root_returns_ok(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_info_returns_paths(self):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "encoders_path" in data


class TestPredictEndpoint:
    def test_valid_predict_returns_200(self):
        payload = {**VALID_STUDENT, **VALID_TASK}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_structure(self):
        payload = {**VALID_STUDENT, **VALID_TASK}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "success_probability" in data
        assert "prediction" in data
        assert "confidence_level" in data

    def test_predict_probability_range(self):
        payload = {**VALID_STUDENT, **VALID_TASK}
        response = client.post("/predict", json=payload)
        prob = response.json()["success_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_valid_labels(self):
        payload = {**VALID_STUDENT, **VALID_TASK}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert data["prediction"] in ["Exitoso", "No Exitoso"]
        assert data["confidence_level"] in ["Alta", "Media", "Baja"]

    def test_predict_invalid_skill_range(self):
        payload = {**VALID_STUDENT, **VALID_TASK, "backend_skill": 10}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_task_type(self):
        payload = {**VALID_STUDENT, **VALID_TASK, "task_type": "DesignUI"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestRecommendEndpoint:
    def test_valid_recommend_returns_200(self):
        payload = {
            "task": VALID_TASK,
            "students": [
                {"name": "Luis", **VALID_STUDENT},
                {"name": "Ana", **{**VALID_STUDENT, "backend_skill": 2}},
            ],
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 200

    def test_recommend_response_structure(self):
        payload = {
            "task": VALID_TASK,
            "students": [
                {"name": "Luis", **VALID_STUDENT},
                {"name": "Ana", **{**VALID_STUDENT, "backend_skill": 2}},
            ],
        }
        response = client.post("/recommend", json=payload)
        data = response.json()
        assert "recommended_student" in data
        assert "success_probability" in data
        assert "ranking" in data
        assert len(data["ranking"]) == 2

    def test_recommend_ranking_order(self):
        payload = {
            "task": VALID_TASK,
            "students": [
                {"name": "Expert", **{**VALID_STUDENT, "backend_skill": 5, "projects_completed": 15}},
                {"name": "Novice", **{**VALID_STUDENT, "backend_skill": 1, "projects_completed": 0}},
            ],
        }
        response = client.post("/recommend", json=payload)
        data = response.json()
        assert data["recommended_student"] == "Expert"
        probs = [r["success_probability"] for r in data["ranking"]]
        assert probs == sorted(probs, reverse=True)

    def test_recommend_empty_students_returns_422(self):
        payload = {"task": VALID_TASK, "students": []}
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422
