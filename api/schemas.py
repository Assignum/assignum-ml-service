"""
Esquemas Pydantic para validación de requests y responses de la API.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Literal


_CAMEL = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ─── Request schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_config = _CAMEL

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


class StudentInput(BaseModel):
    model_config = _CAMEL

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
    model_config = _CAMEL

    task_type: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    task_complexity: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    estimated_hours: int = Field(..., ge=1, le=200)


class RecommendRequest(BaseModel):
    model_config = _CAMEL

    task: TaskInput
    students: list[StudentInput] = Field(..., min_length=1, max_length=50)


class NewTaskRecord(BaseModel):
    model_config = _CAMEL

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


class TaskItemInput(BaseModel):
    model_config = _CAMEL

    task_id: str = Field(..., description="Identificador unico de la tarea")
    task_name: str = Field(..., description="Nombre descriptivo de la tarea")
    task_type: Literal[
        "Backend", "Frontend", "Testing", "Database", "Documentation", "Management"
    ]
    task_complexity: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    estimated_hours: int = Field(..., ge=1, le=200)


class AssignRequest(BaseModel):
    model_config = _CAMEL

    tasks: list[TaskItemInput] = Field(..., min_length=1, max_length=100)
    students: list[StudentInput] = Field(..., min_length=1, max_length=50)


class TaskAssignmentResponse(BaseModel):
    task_id: str
    task_name: str
    assigned_to: str
    success_probability: float
    estimated_hours: int
    task_type: str
    task_complexity: str
    priority: str


class WorkloadSummaryResponse(BaseModel):
    student_name: str
    assigned_tasks: list[str]
    total_hours: int
    available_hours: int
    utilization_pct: float


class AssignResponse(BaseModel):
    assignments: list[TaskAssignmentResponse]
    workload: list[WorkloadSummaryResponse]


class AppendDataResponse(BaseModel):
    message: str
    total_records: int


class RetrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    message: str
    model_updated: bool
    metrics: dict | None = None
