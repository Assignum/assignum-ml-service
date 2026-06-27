"""
Capa de servicio: carga el modelo y los encoders una sola vez (singleton)
y expone funciones para predecir y recomendar.
"""

from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from src.training import load_model
from src.preprocessing import load_encoders, encode_single_record
from src.recommendation import StudentProfile, TaskProfile, rank_candidates, CandidateResult
from api.schemas import PredictRequest, StudentInput, TaskInput


@lru_cache(maxsize=1)
def _get_model() -> RandomForestClassifier:
    """Carga el modelo una única vez y lo cachea en memoria."""
    return load_model()


@lru_cache(maxsize=1)
def _get_encoders() -> dict:
    """Carga los encoders una única vez y los cachea."""
    return load_encoders()


def predict_single(request: PredictRequest) -> tuple[float, str, str]:
    """
    Predice la probabilidad de éxito para un par (estudiante, tarea).

    Returns:
        (success_probability, prediction_label, confidence_level)
    """
    model = _get_model()
    encoders = _get_encoders()

    record = request.model_dump()
    X = encode_single_record(record, encoders)

    prob = float(model.predict_proba(X)[0][1])
    prediction = "Exitoso" if prob >= 0.5 else "No Exitoso"

    if prob >= 0.75 or prob <= 0.25:
        confidence = "Alta"
    elif prob >= 0.60 or prob <= 0.40:
        confidence = "Media"
    else:
        confidence = "Baja"

    return prob, prediction, confidence


def recommend_for_task(
    task_input: TaskInput,
    students_input: list[StudentInput],
) -> list[CandidateResult]:
    """
    Genera el ranking de candidatos para una tarea dada.
    """
    model = _get_model()
    encoders = _get_encoders()

    task = TaskProfile(
        task_type=task_input.task_type,
        task_complexity=task_input.task_complexity,
        priority=task_input.priority,
        estimated_hours=task_input.estimated_hours,
    )

    students = [
        StudentProfile(**s.model_dump())
        for s in students_input
    ]

    return rank_candidates(task, students, model, encoders)


def reload_model() -> None:
    """Limpia el caché para forzar la recarga del modelo desde disco."""
    _get_model.cache_clear()
    _get_encoders.cache_clear()
    print("Caché del modelo limpiado. Se recargará en la próxima predicción.")
