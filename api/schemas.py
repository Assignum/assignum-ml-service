"""
Esquemas Pydantic para validación de requests y responses de la API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ─── Request schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Payload para el endpoint POST /predict.
    Combina el perfil del estudiante y las características de la tarea.
    """

    # Competencias técnicas
    backend_skill: int = Field(..., ge=1, le=5, description="Habilidad backend (1-5)")
    frontend_skill: int = Field(..., ge=1, le=5)
    database_skill: int = Field(..., ge=1, le=5)
    testing_skill: int = Field(..., ge=1, le=5)
    documentation_skill: int = Field(..., ge=1, le=5)
    git_github_skill: int = Field(..., ge=1, le=5)
    agile_methodologies_skill: int = Field(..., ge=1, le=5)

    # Competencias colaborativas
    teamwork_skill: int = Field(..., ge=1, le=5)
    communication_skill: int = Field(..., ge=1, le=5)
    leadership_skill: int = Field(..., ge=1, le=5)
    organization_skill: int = Field(..., ge=1, le=5)

    # Experiencia
    projects_completed: int = Field(..., ge=0)
    available_hours_per_week: int = Field(..., ge=1, le=80)
    last_role: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    last_role_performance: int = Field(..., ge=1, le=5)
    peer_evaluation: int = Field(..., ge=1, le=5)

    # Características de la tarea
    task_type: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    task_complexity: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    estimated_hours: int = Field(..., ge=1, le=200)

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "task_type": "Backend",
                "task_complexity": "Medium",
                "priority": "High",
                "estimated_hours": 15,
            }
        }
    }


class StudentInput(BaseModel):
    """Perfil de estudiante para el endpoint de recomendación."""

    name: str = Field(..., min_length=1, max_length=100)
    backend_skill: int = Field(3, ge=1, le=5)
    frontend_skill: int = Field(3, ge=1, le=5)
    database_skill: int = Field(3, ge=1, le=5)
    testing_skill: int = Field(3, ge=1, le=5)
    documentation_skill: int = Field(3, ge=1, le=5)
    git_github_skill: int = Field(3, ge=1, le=5)
    agile_methodologies_skill: int = Field(3, ge=1, le=5)
    teamwork_skill: int = Field(3, ge=1, le=5)
    communication_skill: int = Field(3, ge=1, le=5)
    leadership_skill: int = Field(3, ge=1, le=5)
    organization_skill: int = Field(3, ge=1, le=5)
    projects_completed: int = Field(0, ge=0)
    available_hours_per_week: int = Field(10, ge=1, le=80)
    last_role: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ] = "Backend"
    last_role_performance: int = Field(3, ge=1, le=5)
    peer_evaluation: int = Field(3, ge=1, le=5)


class TaskInput(BaseModel):
    """Descripción de la tarea para el endpoint de recomendación."""

    task_type: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    task_complexity: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    estimated_hours: int = Field(..., ge=1, le=200)


class RecommendRequest(BaseModel):
    """Payload para POST /recommend."""

    task: TaskInput
    students: list[StudentInput] = Field(..., min_length=1, max_length=50)

    model_config = {
        "json_schema_extra": {
            "example": {
                "task": {
                    "task_type": "Backend",
                    "task_complexity": "Medium",
                    "priority": "High",
                    "estimated_hours": 15,
                },
                "students": [
                    {"name": "Luis", "backend_skill": 5, "available_hours_per_week": 20},
                    {"name": "Ana", "backend_skill": 3, "available_hours_per_week": 15},
                ],
            }
        }
    }


class NewTaskRecord(BaseModel):
    """Registro de una tarea completada para reentrenamiento."""

    backend_skill: int = Field(..., ge=1, le=5)
    frontend_skill: int = Field(..., ge=1, le=5)
    database_skill: int = Field(..., ge=1, le=5)
    testing_skill: int = Field(..., ge=1, le=5)
    documentation_skill: int = Field(..., ge=1, le=5)
    git_github_skill: int = Field(..., ge=1, le=5)
    agile_methodologies_skill: int = Field(..., ge=1, le=5)
    teamwork_skill: int = Field(..., ge=1, le=5)
    communication_skill: int = Field(..., ge=1, le=5)
    leadership_skill: int = Field(..., ge=1, le=5)
    organization_skill: int = Field(..., ge=1, le=5)
    projects_completed: int = Field(..., ge=0)
    available_hours_per_week: int = Field(..., ge=1, le=80)
    last_role: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    last_role_performance: int = Field(..., ge=1, le=5)
    peer_evaluation: int = Field(..., ge=1, le=5)
    task_type: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    task_complexity: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    estimated_hours: int = Field(..., ge=1)
    success: Literal[0, 1]


# ─── Response schemas ─────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    success_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: Literal["Exitoso", "No Exitoso"]
    confidence_level: Literal["Alta", "Media", "Baja"]


class CandidateResponse(BaseModel):
    name: str
    success_probability: float
    prediction: Literal["Exitoso", "No Exitoso"]
    rank: int


class RecommendResponse(BaseModel):
    recommended_student: str
    success_probability: float
    ranking: list[CandidateResponse]


class AppendDataResponse(BaseModel):
    message: str
    total_records: int


class RetrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    message: str
    model_updated: bool
    metrics: dict | None = None
