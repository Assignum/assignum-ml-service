"""
Pipeline principal de entrenamiento.

Uso:
    python main.py                 # Ejecuta pipeline completo + demo
    python main.py --demo          # Solo ejecuta demo de recomendacion
    python main.py --fast          # GridSearch reducido (mas rapido)
    python main.py --samples 2000  # Genera dataset con N registros
"""

import argparse
from pathlib import Path


def run_training_pipeline(n_samples: int = 1300, fast_mode: bool = True) -> None:
    print("\n" + "=" * 60)
    print("  ASSIGNUM ML — PIPELINE DE ENTRENAMIENTO")
    print("=" * 60)

    print("\n[1/6] Generando dataset sintetico...")
    from src.data_generation import generate_dataset, save_dataset
    df = generate_dataset(n_samples=n_samples)
    raw_path = Path("data/raw/synthetic_dataset.csv")
    save_dataset(df, raw_path)

    print("\n[2/6] Preprocesando datos...")
    from src.preprocessing import preprocess_pipeline
    X_train, X_test, y_train, y_test, encoders = preprocess_pipeline(df)

    print("\n[3/6] Entrenando RandomForestClassifier con GridSearchCV...")
    from src.training import train_with_grid_search, run_cross_validation, save_model, get_feature_importances
    model, grid_search = train_with_grid_search(X_train, y_train, fast_mode=fast_mode)

    print("\n[4/6] Ejecutando Cross Validation...")
    run_cross_validation(model, X_train, y_train)

    print("\n[5/6] Evaluando en test set...")
    from src.evaluation import full_evaluation_report
    importance_df = get_feature_importances(model, X_train.columns.tolist())
    metrics = full_evaluation_report(model, X_test, y_test, importance_df, encoders=encoders)

    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(processed_path / "feature_importance.csv", index=False)

    print("\n[6/6] Guardando modelo...")
    save_model(model)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignum ML Pipeline")
    parser.add_argument("--demo", action="store_true", help="Solo ejecuta el demo de recomendacion")
    parser.add_argument("--fast", action="store_true", help="Usa GridSearch reducido")
    parser.add_argument("--samples", type=int, default=1300, help="Numero de registros sinteticos")
    args = parser.parse_args()

    if args.demo:
        from demo_casos import run_all_cases
        run_all_cases()
    else:
        run_training_pipeline(n_samples=args.samples, fast_mode=args.fast)
        from demo_casos import run_all_cases
        run_all_cases()


if __name__ == "__main__":
    main()
