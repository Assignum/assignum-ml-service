"""
Preprocesamiento del dataset: limpieza, validación, encoding y split train/test.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

CATEGORICAL_COLS = ["last_role", "task_type", "task_complexity", "priority"]

SKILL_COLS = [
    "backend_skill", "frontend_skill", "database_skill", "testing_skill",
    "documentation_skill", "git_github_skill", "agile_methodologies_skill",
    "teamwork_skill", "communication_skill", "leadership_skill", "organization_skill",
    "last_role_performance", "peer_evaluation",
]

NUMERIC_COLS = ["projects_completed", "available_hours_per_week", "estimated_hours"]

TARGET_COL = "success"

VALID_TASK_TYPES = ["Backend", "Frontend", "Testing", "Database", "Documentation", "Management"]
VALID_COMPLEXITIES = ["Low", "Medium", "High"]
VALID_PRIORITIES = ["Low", "Medium", "High"]
VALID_ROLES = ["Backend", "Frontend", "Testing", "Database", "Documentation", "Management"]

ENCODERS_PATH = Path(__file__).parent.parent / "models" / "label_encoders.joblib"


TASK_SKILL_MAP = {
    "Backend": "backend_skill",
    "Frontend": "frontend_skill",
    "Testing": "testing_skill",
    "Database": "database_skill",
    "Documentation": "documentation_skill",
    "Management": "leadership_skill",
}

COMPLEXITY_ORDER = {"Low": 1, "Medium": 2, "High": 3}
PRIORITY_ORDER = {"Low": 1, "Medium": 2, "High": 3}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de ingeniería que capturan directamente la compatibilidad
    estudiante-tarea. Esto evita que el modelo deba aprender interacciones
    implícitas entre task_type y las skills individuales.

    Features creadas:
      - relevant_skill: habilidad específica del estudiante para el tipo de tarea
      - skill_task_match: diferencia entre relevant_skill y complejidad (en escala 1-5)
      - collab_avg: promedio de habilidades colaborativas
      - avail_complexity_ratio: disponibilidad relativa a complejidad
      - role_matches_task: 1 si el rol previo coincide con el tipo de tarea
    """
    df = df.copy()

    # Relevant skill para el tipo de tarea
    df["relevant_skill"] = df.apply(
        lambda r: r[TASK_SKILL_MAP.get(r["task_type"], "backend_skill")]
        if r["task_type"] in TASK_SKILL_MAP else 3,
        axis=1,
    ).astype(float)

    # Fit entre skill y complejidad: skill alta + complejidad baja = más apto
    complexity_num = df["task_complexity"].map(COMPLEXITY_ORDER).fillna(2)
    df["skill_complexity_gap"] = df["relevant_skill"] - complexity_num * 0.8

    # Promedio de habilidades colaborativas
    df["collab_avg"] = df[["teamwork_skill", "communication_skill", "organization_skill"]].mean(axis=1)

    # Disponibilidad relativa a horas estimadas
    df["avail_hours_ratio"] = (df["available_hours_per_week"] / df["estimated_hours"].clip(lower=1)).clip(upper=5)

    # Coincidencia de rol previo con tipo de tarea
    df["role_matches_task"] = (df["last_role"] == df["task_type"]).astype(int)

    # Score compuesto de factores secundarios (normalizado 0-1).
    # Captura la ventaja acumulada de experiencia, disponibilidad, evaluación
    # y coincidencia de rol — permite diferenciar candidatos con misma skill
    # principal pero distinto perfil de soporte.
    exp_norm = (df["projects_completed"].clip(upper=12) / 12.0)
    avail_norm = (df["available_hours_per_week"].clip(upper=40) / 40.0)
    peer_norm = (df["peer_evaluation"] - 1) / 4.0
    perf_norm = (df["last_role_performance"] - 1) / 4.0
    df["secondary_score"] = (
        exp_norm   * 0.30
        + avail_norm * 0.25
        + peer_norm  * 0.25
        + perf_norm  * 0.10
        + df["role_matches_task"] * 0.10
    )

    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y limpia el DataFrame:
    - Elimina duplicados.
    - Rellena nulos en skills con la mediana.
    - Filtra valores fuera de rango.
    - Valida categorías.
    """
    initial_rows = len(df)

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Rellenar nulos en columnas de skills con la mediana
    for col in SKILL_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Rellenar nulos en numéricos con la mediana
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Rellenar nulos en categóricas con la moda
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clamp skills al rango válido [1, 5]
    for col in SKILL_COLS:
        if col in df.columns:
            df[col] = df[col].clip(1, 5)

    # Filtrar valores no válidos en categóricas
    if "task_type" in df.columns:
        df = df[df["task_type"].isin(VALID_TASK_TYPES)]
    if "task_complexity" in df.columns:
        df = df[df["task_complexity"].isin(VALID_COMPLEXITIES)]
    if "priority" in df.columns:
        df = df[df["priority"].isin(VALID_PRIORITIES)]
    if "last_role" in df.columns:
        df = df[df["last_role"].isin(VALID_ROLES)]

    # Clamp valores numéricos
    if "projects_completed" in df.columns:
        df["projects_completed"] = df["projects_completed"].clip(0, 50)
    if "available_hours_per_week" in df.columns:
        df["available_hours_per_week"] = df["available_hours_per_week"].clip(1, 80)
    if "estimated_hours" in df.columns:
        df["estimated_hours"] = df["estimated_hours"].clip(1, 100)

    removed = initial_rows - len(df)
    if removed > 0:
        print(f"Validación: {removed} registros eliminados.")

    df = df.reset_index(drop=True)
    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Aplica Label Encoding a las columnas categóricas.

    Args:
        df: DataFrame de entrada.
        fit: Si True, ajusta nuevos encoders. Si False, usa los encoders provistos.
        encoders: Diccionario de encoders pre-entrenados (requerido si fit=False).

    Returns:
        (df_encoded, encoders)
    """
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        if encoders is None:
            raise ValueError("Se requieren encoders pre-entrenados cuando fit=False.")
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = encoders[col]
                df[col] = le.transform(df[col].astype(str))

    return df, encoders


ENGINEERED_COLS = [
    "relevant_skill",
    "skill_complexity_gap",
    "collab_avg",
    "avail_hours_ratio",
    "role_matches_task",
    "secondary_score",
]


def get_feature_columns() -> list[str]:
    """Retorna la lista ordenada de columnas de features (excluyendo el target)."""
    return SKILL_COLS + NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_COLS


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    save_encoders: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, LabelEncoder]]:
    """
    Pipeline completo de preprocesamiento.

    Returns:
        X_train, X_test, y_train, y_test, encoders
    """
    df = validate_dataframe(df)
    # Features engineered ANTES del encoding (usan las columnas categóricas originales)
    df = add_engineered_features(df)
    df, encoders = encode_categoricals(df, fit=True)

    if save_encoders:
        ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)
        print(f"Encoders guardados en: {ENCODERS_PATH}")

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    print(f"Distribución train - éxito: {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test, encoders


def load_encoders() -> dict[str, LabelEncoder]:
    """Carga los encoders persistidos desde disco."""
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron encoders en {ENCODERS_PATH}. "
            "Ejecute el pipeline de entrenamiento primero."
        )
    return joblib.load(ENCODERS_PATH)


def encode_single_record(record: dict, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Codifica un único registro (diccionario) para usarlo con el modelo en producción.
    Retorna un DataFrame con las columnas en el orden correcto.
    """
    df = pd.DataFrame([record])
    df = add_engineered_features(df)
    df, _ = encode_categoricals(df, fit=False, encoders=encoders)
    feature_cols = get_feature_columns()
    return df[feature_cols]
