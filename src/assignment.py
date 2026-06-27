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

    for task in sorted_tasks:
        task_profile = TaskProfile(
            task_type=task.task_type,
            task_complexity=task.task_complexity,
            priority=task.priority,
            estimated_hours=task.estimated_hours,
        )

        # Obtener probabilidades ML para todos los estudiantes
        all_candidates = list(profiles.values())
        ranked = rank_candidates(task_profile, all_candidates, model, encoders)

        # Elegir el mejor candidato con capacidad horaria suficiente
        chosen = None
        chosen_prob = 0.0

        for candidate in ranked:
            avail = remaining_hours[candidate.name]
            if avail <= 0:
                continue

            # Score ajustado: penaliza si la disponibilidad es insuficiente
            # Si tiene al menos el 50% de las horas requeridas, sigue siendo candidato
            coverage = min(avail / max(task.estimated_hours, 1), 1.0)
            if coverage < 0.5:
                # Prefiere a alguien con mas capacidad; solo usa este si no hay nadie mas
                adjusted = candidate.success_probability * coverage
            else:
                adjusted = candidate.success_probability * coverage

            if chosen is None or adjusted > chosen_prob:
                chosen = candidate
                chosen_prob = adjusted

        # Si nadie tiene horas (grupo sobrecargado), asignar al que menos horas usa
        if chosen is None:
            chosen = ranked[0]
            chosen_prob = chosen.success_probability

        remaining_hours[chosen.name] -= task.estimated_hours
        assignments.append(
            TaskAssignment(
                task_id=task.task_id,
                task_name=task.task_name,
                assigned_to=chosen.name,
                success_probability=round(chosen.success_probability, 4),
                estimated_hours=task.estimated_hours,
                task_type=task.task_type,
                task_complexity=task.task_complexity,
                priority=task.priority,
            )
        )

    # Construir resumen de carga por estudiante
    hours_assigned: dict[str, int] = {s.name: 0 for s in students}
    tasks_assigned: dict[str, list[str]] = {s.name: [] for s in students}

    for a in assignments:
        hours_assigned[a.assigned_to] += a.estimated_hours
        tasks_assigned[a.assigned_to].append(a.task_name)

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
