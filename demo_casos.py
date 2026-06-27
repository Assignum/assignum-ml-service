"""
Casos de prueba para evaluar la precisión del sistema de recomendación.

Cada caso está diseñado para ser "reñido" pero con un ganador lógicamente claro,
permitiendo validar que el modelo aprende las reglas del dominio correctamente.
"""

from src.recommendation import StudentProfile, TaskProfile, rank_candidates, print_recommendation_report
from src.preprocessing import load_encoders
from src.training import load_model


def run_all_cases() -> None:
    model = load_model()
    encoders = load_encoders()

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


if __name__ == "__main__":
    run_all_cases()
