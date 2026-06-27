"""
Casos de prueba para evaluar el sistema de recomendacion individual
y el sistema de asignacion grupal de multiples tareas.

Recomendacion: casos reñidos con un ganador logicamente claro.
Asignacion:    sprints con N tareas y M estudiantes — valida equidad y aptitud.
"""

from src.recommendation import StudentProfile, TaskProfile, rank_candidates, print_recommendation_report
from src.assignment import assign_tasks, TaskItem, StudentItem
from src.preprocessing import load_encoders
from src.training import load_model


def run_all_cases() -> None:
    model = load_model()
    encoders = load_encoders()

    # ── Casos de recomendacion individual ────────────────────────────────────
    casos = [
        _caso_1_mismo_skill_diferente_disponibilidad(),
        _caso_2_mismo_skill_diferente_experiencia_y_rol(),
        _caso_3_management_liderazgo_vs_skill_tecnica(),
        _caso_4_skill4_disponibilidad_extrema(),
        _caso_5_tres_con_skill3_factores_secundarios_deciden(),
    ]

    for tarea, estudiantes, descripcion in casos:
        print(f"\n{'='*60}")
        print(f"  {descripcion}")
        print(f"{'='*60}")
        resultados = rank_candidates(tarea, estudiantes, model, encoders)
        print_recommendation_report(tarea, resultados)

    # ── Casos de asignacion grupal ────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print(f"  ASIGNACION GRUPAL — DISTRIBUCION DE SPRINTS")
    print(f"{'#'*60}")

    casos_asignacion = [
        _asignacion_1_sprint_completo_roles_claros(),
        _asignacion_2_carga_equitativa_misma_habilidad(),
        _asignacion_3_tareas_simples_para_menos_capaces(),
        _asignacion_4_grupo_sobrecargado(),
        _asignacion_5_sprint_mixto_cuatro_integrantes(),
    ]

    for tareas, estudiantes, descripcion in casos_asignacion:
        print(f"\n{'='*60}")
        print(f"  {descripcion}")
        print(f"{'='*60}")
        assignments, workload = assign_tasks(tareas, estudiantes, model, encoders)
        _print_assignment_report(assignments, workload)


def _caso_1_mismo_skill_diferente_disponibilidad():
    """
    Caso 1: Mismo backend_skill=4, diferente disponibilidad en tarea High.
    Ganador esperado: Marco (35h disponibles vs 5h de Paola).
    Razonamiento: tarea High con 30h estimadas necesita disponibilidad real.
    """
    tarea = TaskProfile(
        task_type="Backend",
        task_complexity="High",
        priority="High",
        estimated_hours=30,
    )
    estudiantes = [
        StudentProfile(
            name="Marco",
            backend_skill=4, frontend_skill=3, database_skill=3,
            testing_skill=3, documentation_skill=2, git_github_skill=4,
            agile_methodologies_skill=3, teamwork_skill=4, communication_skill=3,
            leadership_skill=2, organization_skill=3, projects_completed=6,
            available_hours_per_week=35, last_role="Backend",
            last_role_performance=4, peer_evaluation=4,
        ),
        StudentProfile(
            name="Paola",
            backend_skill=4, frontend_skill=3, database_skill=3,
            testing_skill=3, documentation_skill=2, git_github_skill=4,
            agile_methodologies_skill=3, teamwork_skill=4, communication_skill=3,
            leadership_skill=2, organization_skill=3, projects_completed=6,
            available_hours_per_week=5, last_role="Backend",
            last_role_performance=4, peer_evaluation=4,
        ),
    ]
    return tarea, estudiantes, "CASO 1 — Mismo skill=4, Marco(35h) vs Paola(5h). Gana Marco."


def _caso_2_mismo_skill_diferente_experiencia_y_rol():
    """
    Caso 2: Ambos frontend_skill=3 (zona media), pero Sofia tiene todo a favor
    y Diego tiene todo en contra.
    Ganador esperado: Sofia (rol match + experiencia + peer eval alta).
    """
    tarea = TaskProfile(
        task_type="Frontend",
        task_complexity="Medium",
        priority="Medium",
        estimated_hours=15,
    )
    estudiantes = [
        StudentProfile(
            name="Sofia",
            backend_skill=2, frontend_skill=3, database_skill=2,
            testing_skill=2, documentation_skill=3, git_github_skill=3,
            agile_methodologies_skill=3, teamwork_skill=4, communication_skill=4,
            leadership_skill=3, organization_skill=4, projects_completed=8,
            available_hours_per_week=20, last_role="Frontend",
            last_role_performance=5, peer_evaluation=5,
        ),
        StudentProfile(
            name="Diego",
            backend_skill=4, frontend_skill=3, database_skill=4,
            testing_skill=3, documentation_skill=2, git_github_skill=3,
            agile_methodologies_skill=2, teamwork_skill=2, communication_skill=2,
            leadership_skill=2, organization_skill=2, projects_completed=1,
            available_hours_per_week=8, last_role="Backend",
            last_role_performance=2, peer_evaluation=2,
        ),
    ]
    return tarea, estudiantes, "CASO 2 — Ambos frontend_skill=3. Sofia(rol+exp+peer) vs Diego(sin match). Gana Sofia."


def _caso_3_management_liderazgo_vs_skill_tecnica():
    """
    Caso 3: Tarea de Management. Valeria tiene liderazgo alto pero skill tecnica
    baja. Roberto tiene skill tecnica alta pero poco liderazgo.
    Ganador esperado: Valeria (tarea Management evalua leadership_skill).
    Valida que el modelo usa la skill correcta segun el tipo de tarea.
    """
    tarea = TaskProfile(
        task_type="Management",
        task_complexity="Medium",
        priority="High",
        estimated_hours=12,
    )
    estudiantes = [
        StudentProfile(
            name="Valeria",
            backend_skill=2, frontend_skill=2, database_skill=2,
            testing_skill=2, documentation_skill=3, git_github_skill=2,
            agile_methodologies_skill=4, teamwork_skill=5, communication_skill=5,
            leadership_skill=5, organization_skill=5, projects_completed=7,
            available_hours_per_week=20, last_role="Management",
            last_role_performance=5, peer_evaluation=5,
        ),
        StudentProfile(
            name="Roberto",
            backend_skill=5, frontend_skill=5, database_skill=5,
            testing_skill=4, documentation_skill=4, git_github_skill=5,
            agile_methodologies_skill=3, teamwork_skill=3, communication_skill=3,
            leadership_skill=2, organization_skill=2, projects_completed=5,
            available_hours_per_week=20, last_role="Backend",
            last_role_performance=4, peer_evaluation=3,
        ),
    ]
    return tarea, estudiantes, "CASO 3 — Management: Valeria(liderazgo=5) vs Roberto(tecnico=5, liderazgo=2). Gana Valeria."


def _caso_4_skill4_disponibilidad_extrema():
    """
    Caso 4: Tres candidatos con database_skill=4, pero diferente disponibilidad
    para una tarea High de 30h. El orden debe seguir la disponibilidad.
    Ganador esperado: Javier (30h) > Elena (15h) > Bruno (4h).
    """
    tarea = TaskProfile(
        task_type="Database",
        task_complexity="High",
        priority="High",
        estimated_hours=30,
    )
    base = dict(
        backend_skill=3, frontend_skill=2, database_skill=4,
        testing_skill=3, documentation_skill=2, git_github_skill=3,
        agile_methodologies_skill=3, teamwork_skill=3, communication_skill=3,
        leadership_skill=2, organization_skill=3, projects_completed=5,
        last_role="Database", last_role_performance=4, peer_evaluation=3,
    )
    estudiantes = [
        StudentProfile(name="Javier", **{**base, "available_hours_per_week": 30}),
        StudentProfile(name="Elena",  **{**base, "available_hours_per_week": 15}),
        StudentProfile(name="Bruno",  **{**base, "available_hours_per_week": 4}),
    ]
    return tarea, estudiantes, "CASO 4 — Mismo perfil, diferente disponibilidad. Javier(30h)>Elena(15h)>Bruno(4h)."


def _caso_5_tres_con_skill3_factores_secundarios_deciden():
    """
    Caso 5: Tres candidatos con testing_skill=3 (zona media).
    Los factores secundarios determinan el ganador.
    Ganador esperado: Laura > Pedro > Fabian.
      Laura: rol match + experiencia alta + buena peer eval
      Pedro: sin rol match pero experiencia media + peer eval media
      Fabian: sin rol match + sin experiencia + peer eval baja
    """
    tarea = TaskProfile(
        task_type="Testing",
        task_complexity="Low",
        priority="Medium",
        estimated_hours=6,
    )
    estudiantes = [
        StudentProfile(
            name="Laura",
            backend_skill=2, frontend_skill=2, database_skill=2,
            testing_skill=3, documentation_skill=3, git_github_skill=3,
            agile_methodologies_skill=3, teamwork_skill=4, communication_skill=4,
            leadership_skill=3, organization_skill=4, projects_completed=9,
            available_hours_per_week=15, last_role="Testing",
            last_role_performance=5, peer_evaluation=5,
        ),
        StudentProfile(
            name="Pedro",
            backend_skill=3, frontend_skill=3, database_skill=3,
            testing_skill=3, documentation_skill=2, git_github_skill=3,
            agile_methodologies_skill=3, teamwork_skill=3, communication_skill=3,
            leadership_skill=2, organization_skill=3, projects_completed=3,
            available_hours_per_week=12, last_role="Backend",
            last_role_performance=2, peer_evaluation=3,
        ),
        StudentProfile(
            name="Fabian",
            backend_skill=3, frontend_skill=4, database_skill=3,
            testing_skill=3, documentation_skill=2, git_github_skill=2,
            agile_methodologies_skill=2, teamwork_skill=2, communication_skill=2,
            leadership_skill=2, organization_skill=2, projects_completed=1,
            available_hours_per_week=10, last_role="Frontend",
            last_role_performance=2, peer_evaluation=2,
        ),
    ]
    return tarea, estudiantes, "CASO 5 — Todos testing_skill=3. Laura(rol+exp+peer5)>Pedro(medio)>Fabian(sin exp+peer2)."


def _print_assignment_report(assignments, workload) -> None:
    print("\nAsignaciones:")
    for a in assignments:
        print(f"  [{a.task_complexity:6}] {a.task_name:30} -> {a.assigned_to:10} ({a.estimated_hours}h, exito={a.success_probability:.0%})")
    print("\nCarga por estudiante:")
    for w in workload:
        barra = "#" * int(w.utilization_pct / 5)
        print(f"  {w.student_name:10} {w.total_hours:3}h/{w.available_hours}h [{barra:<20}] {w.utilization_pct:5.1f}%")
    total_asignado = sum(w.total_hours for w in workload)
    total_disponible = sum(w.available_hours for w in workload)
    print(f"\n  Total sprint: {total_asignado}h de {total_disponible}h disponibles en el grupo")


# ─── Casos de asignacion grupal ──────────────────────────────────────────────

def _asignacion_1_sprint_completo_roles_claros():
    """
    CASO A1: Grupo de 3 con roles bien definidos y tareas que encajan.
    Esperado: cada estudiante recibe principalmente la tarea de su especialidad.
    Luis(backend) -> API, Carlos(database) -> BD, Maria(docs) -> Documentacion.
    La tarea de testing se reparte segun capacidad sobrante.
    """
    tareas = [
        TaskItem("T1", "Implementar API REST",    "Backend",       "High",   "High",   20),
        TaskItem("T2", "Modelar base de datos",   "Database",      "Medium", "High",   12),
        TaskItem("T3", "Documentar endpoints",    "Documentation", "Low",    "Medium",  5),
        TaskItem("T4", "Tests de integracion",    "Testing",       "Low",    "Medium",  4),
    ]
    estudiantes = [
        StudentItem("Luis",  backend_skill=5, testing_skill=3, documentation_skill=2, database_skill=3,
                    available_hours_per_week=25, last_role="Backend", last_role_performance=5, peer_evaluation=5,
                    projects_completed=10, teamwork_skill=4, communication_skill=4, organization_skill=4,
                    leadership_skill=3, git_github_skill=5, agile_methodologies_skill=4, frontend_skill=2),
        StudentItem("Carlos", database_skill=5, backend_skill=3, testing_skill=3, documentation_skill=2,
                    available_hours_per_week=20, last_role="Database", last_role_performance=4, peer_evaluation=4,
                    projects_completed=7, teamwork_skill=3, communication_skill=3, organization_skill=3,
                    leadership_skill=2, git_github_skill=3, agile_methodologies_skill=3, frontend_skill=2),
        StudentItem("Maria", documentation_skill=5, testing_skill=3, backend_skill=1, database_skill=2,
                    available_hours_per_week=15, last_role="Documentation", last_role_performance=5, peer_evaluation=5,
                    projects_completed=6, teamwork_skill=5, communication_skill=5, organization_skill=5,
                    leadership_skill=4, git_github_skill=3, agile_methodologies_skill=3, frontend_skill=3),
    ]
    return tareas, estudiantes, "ASIG-1 — Roles claros. Luis->Backend, Carlos->DB, Maria->Docs."


def _asignacion_2_carga_equitativa_misma_habilidad():
    """
    CASO A2: Tres estudiantes con habilidades similares en Frontend.
    Todas las tareas son del mismo tipo pero de distinto tamano.
    Esperado: el algoritmo distribuye por horas — nadie carga mas del 50%
    del sprint aunque todos tengan la misma aptitud ML.
    """
    tareas = [
        TaskItem("T1", "Pantalla de login",       "Frontend", "Low",    "Medium",  5),
        TaskItem("T2", "Dashboard principal",     "Frontend", "High",   "High",   18),
        TaskItem("T3", "Pantalla de perfil",      "Frontend", "Medium", "Medium",  8),
        TaskItem("T4", "Componente de notif.",    "Frontend", "Low",    "Low",     4),
        TaskItem("T5", "Pantalla de busqueda",    "Frontend", "Medium", "Medium",  7),
    ]
    base = dict(
        frontend_skill=4, backend_skill=2, database_skill=2, testing_skill=2,
        documentation_skill=2, git_github_skill=3, agile_methodologies_skill=3,
        teamwork_skill=3, communication_skill=3, leadership_skill=2, organization_skill=3,
        projects_completed=5, last_role="Frontend", last_role_performance=4, peer_evaluation=4,
    )
    estudiantes = [
        StudentItem("Ana",    **{**base, "available_hours_per_week": 20}),
        StudentItem("Bruno",  **{**base, "available_hours_per_week": 20}),
        StudentItem("Carla",  **{**base, "available_hours_per_week": 20}),
    ]
    return tareas, estudiantes, "ASIG-2 — Misma habilidad, 42h en 60h disponibles. Carga ~14h cada uno."


def _asignacion_3_tareas_simples_para_menos_capaces():
    """
    CASO A3: Un experto (skill=5) y dos junior (skill=2-3) con tareas mixtas.
    Esperado: el experto recibe la tarea compleja (High), los juniors las simples.
    Valida que el modelo no sobrecarga al mejor con tareas que cualquiera puede hacer.
    """
    tareas = [
        TaskItem("T1", "Arquitectura del sistema", "Backend", "High",   "High",   22),
        TaskItem("T2", "CRUD de usuarios",         "Backend", "Low",    "Medium",  6),
        TaskItem("T3", "Validacion de formularios","Backend", "Low",    "Low",     4),
        TaskItem("T4", "Endpoint de busqueda",     "Backend", "Medium", "Medium", 10),
    ]
    estudiantes = [
        StudentItem("Experto", backend_skill=5, available_hours_per_week=30,
                    last_role="Backend", last_role_performance=5, peer_evaluation=5,
                    projects_completed=15, teamwork_skill=4, communication_skill=4,
                    organization_skill=4, leadership_skill=3, git_github_skill=5,
                    agile_methodologies_skill=4, frontend_skill=3, database_skill=4,
                    testing_skill=3, documentation_skill=3),
        StudentItem("Junior1", backend_skill=2, available_hours_per_week=20,
                    last_role="Backend", last_role_performance=3, peer_evaluation=3,
                    projects_completed=1, teamwork_skill=3, communication_skill=3,
                    organization_skill=3, leadership_skill=2, git_github_skill=2,
                    agile_methodologies_skill=2, frontend_skill=2, database_skill=2,
                    testing_skill=2, documentation_skill=2),
        StudentItem("Junior2", backend_skill=2, available_hours_per_week=20,
                    last_role="Frontend", last_role_performance=3, peer_evaluation=3,
                    projects_completed=2, teamwork_skill=3, communication_skill=3,
                    organization_skill=3, leadership_skill=2, git_github_skill=2,
                    agile_methodologies_skill=2, frontend_skill=3, database_skill=2,
                    testing_skill=2, documentation_skill=2),
    ]
    return tareas, estudiantes, "ASIG-3 — Experto(skill=5) + 2 Juniors. Tarea High->Experto, simples->Juniors."


def _asignacion_4_grupo_sobrecargado():
    """
    CASO A4: El sprint tiene mas horas de las que el grupo puede cubrir.
    Esperado: el algoritmo asigna de todas formas (no falla), priorizando
    aptitud cuando ya no hay capacidad libre. El reporte muestra >100% en alguien.
    """
    tareas = [
        TaskItem("T1", "Modulo de pagos",       "Backend",  "High",   "High",   30),
        TaskItem("T2", "Integracion con API",   "Backend",  "High",   "High",   25),
        TaskItem("T3", "Panel de admin",        "Frontend", "Medium", "High",   15),
        TaskItem("T4", "Tests de regresion",    "Testing",  "Medium", "Medium", 12),
    ]
    estudiantes = [
        StudentItem("Ivan",   backend_skill=4, frontend_skill=3, testing_skill=3,
                    available_hours_per_week=20, last_role="Backend",
                    last_role_performance=4, peer_evaluation=4, projects_completed=6,
                    teamwork_skill=3, communication_skill=3, organization_skill=3,
                    leadership_skill=2, git_github_skill=4, agile_methodologies_skill=3,
                    documentation_skill=2, database_skill=3),
        StudentItem("Sofia",  backend_skill=3, frontend_skill=4, testing_skill=3,
                    available_hours_per_week=15, last_role="Frontend",
                    last_role_performance=4, peer_evaluation=4, projects_completed=5,
                    teamwork_skill=4, communication_skill=4, organization_skill=4,
                    leadership_skill=3, git_github_skill=3, agile_methodologies_skill=3,
                    documentation_skill=3, database_skill=2),
    ]
    return tareas, estudiantes, "ASIG-4 — Grupo sobrecargado (82h de tarea, 35h disponibles). Fallback activo."


def _asignacion_5_sprint_mixto_cuatro_integrantes():
    """
    CASO A5: Sprint realista con 4 integrantes y 6 tareas de tipos variados.
    Cada estudiante tiene un perfil distinto. Esperado: distribucion balanceada
    donde cada uno recibe principalmente las tareas que mejor domina,
    sin que nadie supere el 80% de su disponibilidad si hay capacidad de sobra.
    """
    tareas = [
        TaskItem("T1", "Auth con JWT",           "Backend",       "High",   "High",   18),
        TaskItem("T2", "UI de registro",         "Frontend",      "Medium", "High",   10),
        TaskItem("T3", "Esquema de BD",          "Database",      "Medium", "Medium",  8),
        TaskItem("T4", "Manual de usuario",      "Documentation", "Low",    "Low",     4),
        TaskItem("T5", "Tests unitarios",        "Testing",       "Low",    "Medium",  5),
        TaskItem("T6", "Sprint planning",        "Management",    "Low",    "High",    3),
    ]
    estudiantes = [
        StudentItem("Luis",    backend_skill=5, frontend_skill=2, database_skill=3, testing_skill=3,
                    documentation_skill=2, leadership_skill=3, agile_methodologies_skill=4,
                    teamwork_skill=4, communication_skill=4, organization_skill=4,
                    git_github_skill=5, available_hours_per_week=25,
                    last_role="Backend", last_role_performance=5, peer_evaluation=5, projects_completed=12),
        StudentItem("Ana",     frontend_skill=5, backend_skill=2, database_skill=2, testing_skill=3,
                    documentation_skill=3, leadership_skill=3, agile_methodologies_skill=3,
                    teamwork_skill=4, communication_skill=4, organization_skill=4,
                    git_github_skill=3, available_hours_per_week=20,
                    last_role="Frontend", last_role_performance=4, peer_evaluation=4, projects_completed=8),
        StudentItem("Carlos",  database_skill=5, backend_skill=3, frontend_skill=2, testing_skill=3,
                    documentation_skill=2, leadership_skill=2, agile_methodologies_skill=3,
                    teamwork_skill=3, communication_skill=3, organization_skill=3,
                    git_github_skill=3, available_hours_per_week=20,
                    last_role="Database", last_role_performance=4, peer_evaluation=3, projects_completed=7),
        StudentItem("Valeria", leadership_skill=5, agile_methodologies_skill=5, teamwork_skill=5,
                    communication_skill=5, organization_skill=5, backend_skill=2, frontend_skill=3,
                    database_skill=2, testing_skill=3, documentation_skill=4,
                    git_github_skill=3, available_hours_per_week=15,
                    last_role="Management", last_role_performance=5, peer_evaluation=5, projects_completed=9),
    ]
    return tareas, estudiantes, "ASIG-5 — Sprint real: Luis(BE)+Ana(FE)+Carlos(DB)+Valeria(Mgmt). 48h en 80h disponibles."


if __name__ == "__main__":
    run_all_cases()
