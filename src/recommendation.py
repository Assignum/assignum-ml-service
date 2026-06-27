"""
Sistema de recomendación: dado una tarea y una lista de estudiantes,
ordena a los candidatos por probabilidad de éxito.
"""

from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import encode_single_record, load_encoders


@dataclass
class StudentProfile:
    """Perfil completo de un estudiante para el sistema de recomendación."""

    name: str

    # Competencias técnicas (1-5)
    backend_skill: int = 3
    frontend_skill: int = 3
    database_skill: int = 3
    testing_skill: int = 3
    documentation_skill: int = 3
    git_github_skill: int = 3
    agile_methodologies_skill: int = 3

    # Competencias colaborativas (1-5)
    teamwork_skill: int = 3
    communication_skill: int = 3
    leadership_skill: int = 3
    organization_skill: int = 3

    # Experiencia
    projects_completed: int = 0
    available_hours_per_week: int = 10
    last_role: str = "Backend"
    last_role_performance: int = 3
    peer_evaluation: int = 3

    def to_dict(self) -> dict:
        """Convierte el perfil a diccionario, excluyendo el nombre."""
        return {k: v for k, v in self.__dict__.items() if k != "name"}


@dataclass
class TaskProfile:
    """Descripción de una tarea a asignar."""

    task_type: str          # Backend | Frontend | Testing | Database | Documentation | Management
    task_complexity: str    # Low | Medium | High
    priority: str           # Low | Medium | High
    estimated_hours: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class CandidateResult:
    name: str
    success_probability: float
    prediction: str

    def __str__(self) -> str:
        bar = "#" * int(self.success_probability * 20)
        return (
            f"  {self.name:<20} {self.success_probability:>6.1%}  [{bar:<20}]  "
            f"-> {self.prediction}"
        )


def rank_candidates(
    task: TaskProfile,
    students: list[StudentProfile],
    model: RandomForestClassifier,
    encoders: dict | None = None,
) -> list[CandidateResult]:
    """
    Calcula la probabilidad de éxito de cada estudiante para una tarea dada
    y retorna un ranking ordenado de mayor a menor probabilidad.

    Args:
        task: Perfil de la tarea a asignar.
        students: Lista de perfiles de estudiantes candidatos.
        model: Modelo entrenado.
        encoders: Encoders de variables categóricas. Si None, los carga desde disco.

    Returns:
        Lista de CandidateResult ordenada por probabilidad descendente.
    """
    if encoders is None:
        encoders = load_encoders()

    task_data = task.to_dict()
    results: list[CandidateResult] = []

    for student in students:
        record = {**student.to_dict(), **task_data}
        X = encode_single_record(record, encoders)
        prob = float(model.predict_proba(X)[0][1])
        prediction = "Exitoso" if prob >= 0.5 else "No Exitoso"
        results.append(CandidateResult(student.name, prob, prediction))

    results.sort(key=lambda r: r.success_probability, reverse=True)
    return results


def print_recommendation_report(
    task: TaskProfile,
    results: list[CandidateResult],
) -> None:
    """Imprime el ranking de recomendación en formato legible."""
    print("\n" + "=" * 60)
    print("SISTEMA DE RECOMENDACIÓN - ASSIGNUM")
    print("=" * 60)
    print(f"  Tarea:       {task.task_type}")
    print(f"  Complejidad: {task.task_complexity}")
    print(f"  Prioridad:   {task.priority}")
    print(f"  Horas est.:  {task.estimated_hours}h")
    print("-" * 60)
    print("  Ranking de candidatos:")
    print()

    for rank, result in enumerate(results, start=1):
        medal = {1: "[1]", 2: "[2]", 3: "[3]"}.get(rank, f"[{rank}]")
        print(f"  {medal} {result}")

    print()
    best = results[0]
    print(f"  >> RECOMENDACION: Asignar tarea a {best.name} ({best.success_probability:.1%})")
    print("=" * 60)
