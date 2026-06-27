"""
assignment.py
Distribucion equitativa de multiples tareas entre los integrantes de un grupo.

Algoritmo:
  1. Calcula la probabilidad de exito de cada estudiante para cada tarea (ML).
  2. Ordena las tareas de mayor a menor complejidad (horas estimadas).
  3. Asigna cada tarea al estudiante con mayor score ajustado, donde:
       score_ajustado = prob_ml * factor_disponibilidad
       factor_disponibilidad = min(horas_restantes / horas_tarea, 1.0)
  4. Actualiza las horas restantes del estudiante asignado.

Resultado: cada tarea tiene exactamente un asignado, la carga horaria queda
balanceada y las tareas complejas van a los mas capaces que tengan capacidad.
"""

from dataclasses import dataclass, field


@dataclass
class TaskItem:
    task_id: str
    task_name: str
    task_type: str
    task_complexity: str
    priority: str
    estimated_hours: int


@dataclass
class StudentItem:
    name: str
    backend_skill: int = 3
    frontend_skill: int = 3
    database_skill: int = 3
    testing_skill: int = 3
    documentation_skill: int = 3
    git_github_skill: int = 3
    agile_methodologies_skill: int = 3
    teamwork_skill: int = 3
    communication_skill: int = 3
    leadership_skill: int = 3
    organization_skill: int = 3
    projects_completed: int = 0
    available_hours_per_week: int = 10
    last_role: str = "Backend"
    last_role_performance: int = 3
    peer_evaluation: int = 3


@dataclass
class TaskAssignment:
    task_id: str
    task_name: str
    assigned_to: str
    success_probability: float
    estimated_hours: int
    task_type: str
    task_complexity: str
    priority: str


@dataclass
class WorkloadSummary:
    student_name: str
    assigned_tasks: list[str]
    total_hours: int
    available_hours: int
    utilization_pct: float


def assign_tasks(
    tasks: list[TaskItem],
    students: list[StudentItem],
    model,
    encoders: dict,
) -> tuple[list[TaskAssignment], list[WorkloadSummary]]:
    """
    Distribuye una lista de tareas entre los estudiantes del grupo de forma
    equitativa, priorizando aptitud del estudiante y balance de carga horaria.

    Retorna:
        assignments: lista de asignaciones (tarea → estudiante)
        workload:    resumen de carga por estudiante
    """
    from src.recommendation import StudentProfile, TaskProfile, rank_candidates

    # Horas restantes disponibles por estudiante
    remaining_hours: dict[str, int] = {
        s.name: s.available_hours_per_week for s in students
    }

    # Convertir StudentItem → StudentProfile (reutiliza la logica existente)
    profiles = {
        s.name: StudentProfile(
            name=s.name,
            backend_skill=s.backend_skill,
            frontend_skill=s.frontend_skill,
            database_skill=s.database_skill,
            testing_skill=s.testing_skill,
            documentation_skill=s.documentation_skill,
            git_github_skill=s.git_github_skill,
            agile_methodologies_skill=s.agile_methodologies_skill,
            teamwork_skill=s.teamwork_skill,
            communication_skill=s.communication_skill,
            leadership_skill=s.leadership_skill,
            organization_skill=s.organization_skill,
            projects_completed=s.projects_completed,
            available_hours_per_week=s.available_hours_per_week,
            last_role=s.last_role,
            last_role_performance=s.last_role_performance,
            peer_evaluation=s.peer_evaluation,
        )
        for s in students
    }

    # Ordenar tareas: primero las más complejas (más horas estimadas)
    # Así los estudiantes más capaces las reciben antes de quedarse sin capacidad
    sorted_tasks = sorted(tasks, key=lambda t: t.estimated_hours, reverse=True)

    assignments: list[TaskAssignment] = []
    hours_assigned: dict[str, int] = {s.name: 0 for s in students}
    tasks_assigned: dict[str, list[str]] = {s.name: [] for s in students}
    total_sprint_hours = sum(t.estimated_hours for t in tasks)

    for task in sorted_tasks:
        task_profile = TaskProfile(
            task_type=task.task_type,
            task_complexity=task.task_complexity,
            priority=task.priority,
            estimated_hours=task.estimated_hours,
        )

        all_candidates = list(profiles.values())
        ranked = rank_candidates(task_profile, all_candidates, model, encoders)

        # Score final = aptitud ML * equidad de carga
        #
        # equidad = 1 - (horas_ya_asignadas / total_horas_sprint)
        #   → 1.0 si el estudiante no tiene nada asignado (maximo favorecido)
        #   → cercano a 0 si ya tiene la mayoria del sprint asignado
        #
        # Esto balancea activamente: a igual aptitud gana quien tiene menos carga.
        # La aptitud ML sigue pesando: si alguien tiene el doble de aptitud
        # puede recibir mas tareas si la diferencia de habilidad lo justifica.
        best = None
        best_score = -1.0

        for candidate in ranked:
            if remaining_hours[candidate.name] <= 0:
                continue

            carga_relativa = hours_assigned[candidate.name] / max(total_sprint_hours, 1)
            equidad = 1.0 - carga_relativa

            # Penalizacion dura si no alcanza ni el 30% de las horas de la tarea
            cobertura = remaining_hours[candidate.name] / max(task.estimated_hours, 1)
            if cobertura < 0.3:
                equidad *= 0.3

            score = candidate.success_probability * equidad

            if score > best_score:
                best = candidate
                best_score = score

        # Fallback: grupo sobrecargado, asignar al mas apto
        if best is None:
            best = ranked[0]

        hours_assigned[best.name] += task.estimated_hours
        remaining_hours[best.name] -= task.estimated_hours
        tasks_assigned[best.name].append(task.task_name)
        assignments.append(
            TaskAssignment(
                task_id=task.task_id,
                task_name=task.task_name,
                assigned_to=best.name,
                success_probability=round(best.success_probability, 4),
                estimated_hours=task.estimated_hours,
                task_type=task.task_type,
                task_complexity=task.task_complexity,
                priority=task.priority,
            )
        )

    workload = [
        WorkloadSummary(
            student_name=s.name,
            assigned_tasks=tasks_assigned[s.name],
            total_hours=hours_assigned[s.name],
            available_hours=s.available_hours_per_week,
            utilization_pct=round(
                min(hours_assigned[s.name] / max(s.available_hours_per_week, 1), 1.0) * 100, 1
            ),
        )
        for s in students
    ]

    return assignments, workload
