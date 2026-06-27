"""
Pipeline principal de entrenamiento.

Uso:
    python main.py                    # Ejecuta pipeline completo
    python main.py --demo             # Solo ejecuta demo de recomendación
    python main.py --fast             # GridSearch reducido (más rápido)
    python main.py --samples 2000     # Genera dataset con N registros
"""

import argparse
from pathlib import Path


def run_training_pipeline(n_samples: int = 1300, fast_mode: bool = True) -> None:
    print("\n" + "=" * 60)
    print("  ASSIGNUM ML — PIPELINE DE ENTRENAMIENTO")
    print("=" * 60)

    # 1. Generación de datos sintéticos
    print("\n[1/6] Generando dataset sintético...")
    from src.data_generation import generate_dataset, save_dataset
    df = generate_dataset(n_samples=n_samples)
    raw_path = Path("data/raw/synthetic_dataset.csv")
    save_dataset(df, raw_path)

    # 2. Preprocesamiento
    print("\n[2/6] Preprocesando datos...")
    from src.preprocessing import preprocess_pipeline
    X_train, X_test, y_train, y_test, encoders = preprocess_pipeline(df)

    # 3. Entrenamiento con GridSearchCV
    print("\n[3/6] Entrenando RandomForestClassifier con GridSearchCV...")
    from src.training import train_with_grid_search, run_cross_validation, save_model, get_feature_importances
    model, grid_search = train_with_grid_search(X_train, y_train, fast_mode=fast_mode)

    # 4. Cross Validation
    print("\n[4/6] Ejecutando Cross Validation...")
    cv_metrics = run_cross_validation(model, X_train, y_train)

    # 5. Evaluación en test set
    print("\n[5/6] Evaluando en test set...")
    from src.evaluation import full_evaluation_report
    importance_df = get_feature_importances(model, X_train.columns.tolist())
    metrics = full_evaluation_report(model, X_test, y_test, importance_df)

    # Guardar importancias
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(processed_path / "feature_importance.csv", index=False)

    # 6. Persistencia del modelo
    print("\n[6/6] Guardando modelo...")
    save_model(model)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
    print("=" * 60)


def run_recommendation_demo() -> None:
    """Ejecuta 6 casos de prueba para validar el sistema de recomendación."""
    from src.recommendation import StudentProfile, TaskProfile, rank_candidates, print_recommendation_report
    from src.preprocessing import load_encoders
    from src.training import load_model

    model = load_model()
    encoders = load_encoders()

    casos = [
        _caso_original(),
        _caso_1_mismo_skill_diferente_disponibilidad(),
        _caso_2_zona_media_factores_opuestos(),
        _caso_3_management_liderazgo_vs_tecnico(),
        _caso_4_clones_diferente_disponibilidad(),
        _caso_5_zona_media_tres_niveles(),
    ]

    for tarea, estudiantes, descripcion in casos:
        print(f"\n{'='*60}")
        print(f"  {descripcion}")
        resultados = rank_candidates(tarea, estudiantes, model, encoders)
        print_recommendation_report(tarea, resultados)


def _caso_original():
    """Luis(skill=5) vs Carlos(skill=4) vs Ana(skill=3) vs Maria(skill=1). Gana Luis."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Backend", task_complexity="High", priority="High", estimated_hours=25)
    estudiantes = [
        StudentProfile(
            name="Luis",
            backend_skill=5, frontend_skill=2, database_skill=4,
            testing_skill=3, documentation_skill=2, git_github_skill=5,
            agile_methodologies_skill=4, teamwork_skill=4, communication_skill=4,
            leadership_skill=3, organization_skill=4, projects_completed=12,
            available_hours_per_week=30, last_role="Backend",
            last_role_performance=5, peer_evaluation=5,
        ),
        StudentProfile(
            name="Ana",
            backend_skill=3, frontend_skill=5, database_skill=3,
            testing_skill=2, documentation_skill=4, git_github_skill=3,
            agile_methodologies_skill=3, teamwork_skill=5, communication_skill=5,
            leadership_skill=4, organization_skill=5, projects_completed=6,
            available_hours_per_week=20, last_role="Frontend",
            last_role_performance=4, peer_evaluation=4,
        ),
        StudentProfile(
            name="Carlos",
            backend_skill=4, frontend_skill=3, database_skill=4,
            testing_skill=3, documentation_skill=2, git_github_skill=3,
            agile_methodologies_skill=3, teamwork_skill=3, communication_skill=3,
            leadership_skill=2, organization_skill=3, projects_completed=3,
            available_hours_per_week=12, last_role="Database",
            last_role_performance=3, peer_evaluation=3,
        ),
        StudentProfile(
            name="Maria",
            backend_skill=1, frontend_skill=4, database_skill=2,
            testing_skill=2, documentation_skill=5, git_github_skill=2,
            agile_methodologies_skill=4, teamwork_skill=4, communication_skill=4,
            leadership_skill=5, organization_skill=5, projects_completed=2,
            available_hours_per_week=6, last_role="Management",
            last_role_performance=3, peer_evaluation=3,
        ),
    ]
    return tarea, estudiantes, "CASO 0 — Niveles distintos de skill. Gana Luis(skill=5)."


def _caso_1_mismo_skill_diferente_disponibilidad():
    """Mismo backend_skill=4, diferente disponibilidad en tarea High. Gana Marco."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Backend", task_complexity="High", priority="High", estimated_hours=30)
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


def _caso_2_zona_media_factores_opuestos():
    """Ambos frontend_skill=3 (zona media), pero Sofia tiene todo a favor. Gana Sofia."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Frontend", task_complexity="Medium", priority="Medium", estimated_hours=15)
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


def _caso_3_management_liderazgo_vs_tecnico():
    """Tarea Management: liderazgo alto vs skill tecnica alta. Gana Valeria."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Management", task_complexity="Medium", priority="High", estimated_hours=12)
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


def _caso_4_clones_diferente_disponibilidad():
    """Tres perfiles identicos, solo cambia disponibilidad. Javier>Elena>Bruno."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Database", task_complexity="High", priority="High", estimated_hours=30)
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
    return tarea, estudiantes, "CASO 4 — Clones con distinta disponibilidad. Javier(30h)>Elena(15h)>Bruno(4h)."


def _caso_5_zona_media_tres_niveles():
    """Tres con testing_skill=3, factores secundarios deciden el orden. Laura>Pedro>Fabian."""
    from src.recommendation import StudentProfile, TaskProfile
    tarea = TaskProfile(task_type="Testing", task_complexity="Low", priority="Medium", estimated_hours=6)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignum ML Pipeline")
    parser.add_argument("--demo", action="store_true", help="Solo ejecuta el demo de recomendación")
    parser.add_argument("--fast", action="store_true", help="Usa GridSearch reducido")
    parser.add_argument("--samples", type=int, default=1300, help="Número de registros sintéticos")
    args = parser.parse_args()

    if args.demo:
        run_recommendation_demo()
    else:
        run_training_pipeline(n_samples=args.samples, fast_mode=args.fast)
        run_recommendation_demo()


if __name__ == "__main__":
    main()
