"""
Mecanismo de reentrenamiento incremental.

Diseño del ciclo de vida del modelo:
    1. Nuevas tareas completadas se registran en Firebase Firestore (app Flutter).
    2. Un job periódico (cron, Cloud Scheduler, o manual) llama a append_new_data().
    3. Si hay suficientes datos nuevos, retrain_model() reentrenará el modelo.
    4. El nuevo modelo reemplaza al anterior si mejora en la métrica objetivo.

Amenazas a la validez consideradas:
    - Concept drift: el rendimiento de los estudiantes puede cambiar a lo largo
      del semestre. Se recomienda reentrenar al menos una vez por semestre.
    - Data leakage: las evaluaciones de pares solo están disponibles DESPUÉS de
      completar la tarea, por lo que no pueden usarse como feature de predicción
      en tiempo real. La variable peer_evaluation en este modelo representa el
      historial previo del estudiante, no la evaluación actual.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data_generation import TASK_TYPES, COMPLEXITIES, PRIORITIES, ROLES
from src.preprocessing import preprocess_pipeline, validate_dataframe
from src.training import train_with_grid_search, save_model, get_feature_importances
from src.evaluation import evaluate_model

DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "synthetic_dataset.csv"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"
RETRAINING_LOG = PROCESSED_PATH / "retraining_log.json"

# Con 1200 registros sintéticos como base, los datos reales de encuesta
# se agregarán en lotes. 50 registros = ~1 grupo de 5 equipos × 10 tareas.
MIN_NEW_RECORDS = 50


def append_survey_data(survey_records: list[dict]) -> int:
    """
    Incorpora datos reales recopilados de encuestas con estudiantes universitarios.

    Diseño del ciclo de datos de la tesis:
        Fase 1 (actual): 1200 registros sintéticos → modelo base
        Fase 2 (encuesta): +300 registros reales → modelo validado
        Fase 3 (producción): datos continuos de la app Flutter → modelo mejorado

    Los datos de encuesta deben incluir todas las variables del modelo MÁS
    las variables de validación del éxito:
        - completed: bool (tarea entregada)
        - on_time: bool (entregada dentro del plazo)
        - leader_evaluation: int 1-5 (evaluación del líder del equipo)
        - peer_evaluation_post: int 1-5 (evaluación post-tarea de compañeros)

    La variable 'success' se calcula automáticamente como:
        success = 1 si completed AND on_time AND leader_evaluation >= 3
                                    AND peer_evaluation_post >= 3

    Args:
        survey_records: Lista de registros de encuesta con campos de validación.

    Returns:
        Total de registros en el dataset después de la incorporación.
    """
    processed = []
    for record in survey_records:
        # Calcular 'success' desde los criterios reales si no viene calculado
        if "success" not in record:
            completed = record.pop("completed", False)
            on_time = record.pop("on_time", False)
            leader_eval = record.pop("leader_evaluation", 0)
            peer_eval_post = record.pop("peer_evaluation_post", 0)
            record["success"] = int(
                completed and on_time and leader_eval >= 3 and peer_eval_post >= 3
            )
        else:
            # Limpiar campos de validación si ya viene con 'success' calculado
            for field in ["completed", "on_time", "leader_evaluation", "peer_evaluation_post"]:
                record.pop(field, None)
        processed.append(record)

    total = append_new_data(processed)
    print(f"Datos de encuesta incorporados. Dataset total: {total} registros.")
    return total


def append_new_data(new_records: list[dict], dataset_path: Path = DATASET_PATH) -> int:
    """
    Agrega nuevos registros de tareas completadas al dataset histórico.

    Cada registro debe incluir todas las features del estudiante, de la tarea,
    y el resultado real (success: 0 o 1).

    Args:
        new_records: Lista de diccionarios con los datos de nuevas tareas.
        dataset_path: Ruta al CSV del dataset histórico.

    Returns:
        Total de registros en el dataset actualizado.
    """
    df_new = pd.DataFrame(new_records)
    df_new = validate_dataframe(df_new)

    if dataset_path.exists():
        df_existing = pd.read_csv(dataset_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates()
    else:
        df_combined = df_new

    df_combined.to_csv(dataset_path, index=False)
    print(f"Dataset actualizado: {len(df_combined)} registros totales (+{len(df_new)} nuevos)")
    return len(df_combined)


def retrain_model(
    min_new_records: int = MIN_NEW_RECORDS,
    force: bool = False,
    fast_mode: bool = True,
) -> bool:
    """
    Reentrenamiento condicional del modelo.

    Compara el nuevo modelo con el anterior usando ROC AUC en el test set.
    Solo reemplaza el modelo si el nuevo supera al anterior (o si force=True).

    Args:
        min_new_records: Mínimo de registros nuevos desde el último entrenamiento.
        force: Si True, reentrenará sin verificar min_new_records ni comparar métricas.

    Returns:
        True si el modelo fue reemplazado, False en caso contrario.
    """
    if not DATASET_PATH.exists():
        print("No existe dataset. Ejecute la generación de datos primero.")
        return False

    df = pd.read_csv(DATASET_PATH)

    if not force:
        last_count = _get_last_training_count()
        new_records = len(df) - last_count
        if new_records < min_new_records:
            print(
                f"Solo hay {new_records} registros nuevos. "
                f"Se requieren {min_new_records} para reentrenar."
            )
            return False

    print(f"Iniciando reentrenamiento con {len(df)} registros...")

    X_train, X_test, y_train, y_test, encoders = preprocess_pipeline(df, save_encoders=True)
    new_model, _ = train_with_grid_search(X_train, y_train, fast_mode=fast_mode)
    new_metrics = evaluate_model(new_model, X_test, y_test)

    # Comparar con el modelo anterior si existe
    from src.training import load_model, MODEL_PATH

    replace = True
    if MODEL_PATH.exists() and not force:
        try:
            old_model = load_model()
            old_metrics = evaluate_model(old_model, X_test, y_test)
            if new_metrics["roc_auc"] > old_metrics["roc_auc"]:
                print(
                    f"Nuevo modelo MEJOR: ROC AUC {new_metrics['roc_auc']:.4f} "
                    f"> {old_metrics['roc_auc']:.4f}"
                )
            else:
                print(
                    f"Nuevo modelo NO mejora: ROC AUC {new_metrics['roc_auc']:.4f} "
                    f"<= {old_metrics['roc_auc']:.4f}. Manteniendo modelo anterior."
                )
                replace = False
        except Exception:
            replace = True

    if replace:
        save_model(new_model)
        feature_names = X_train.columns.tolist()
        importance_df = get_feature_importances(new_model, feature_names)
        importance_df.to_csv(PROCESSED_PATH / "feature_importance.csv", index=False)
        _log_retraining(len(df), new_metrics)
        print("Modelo reemplazado exitosamente.")

    return replace


def _get_last_training_count() -> int:
    if RETRAINING_LOG.exists():
        with open(RETRAINING_LOG) as f:
            log = json.load(f)
        return log[-1]["total_records"] if log else 0
    return 0


def _log_retraining(total_records: int, metrics: dict) -> None:
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    log = []
    if RETRAINING_LOG.exists():
        with open(RETRAINING_LOG) as f:
            log = json.load(f)

    log.append({
        "timestamp": datetime.now().isoformat(),
        "total_records": total_records,
        "metrics": metrics,
    })

    with open(RETRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log de reentrenamiento actualizado: {RETRAINING_LOG}")
