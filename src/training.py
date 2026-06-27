"""
training.py
Entrenamiento del RandomForestClassifier con GridSearchCV y Cross Validation.

Justificación de Random Forest sobre alternativas:
    - Vs. Árbol de Decisión: RF mitiga el overfitting mediante bagging (promedio de
      múltiples árboles con subconjuntos de features), reduce la varianza sin
      incrementar el sesgo significativamente (Breiman, 2001).
    - Vs. KNN: RF no requiere normalización de features, maneja bien datos mixtos
      (categóricos + numéricos codificados), es más robusto a outliers, y provee
      importancia de features nativa — esencial para justificar las variables en
      el contexto de la tesis.
    - RF es interpretable mediante Feature Importance (MDI), compatible con
      explicaciones académicas y con el marco de competencias del SINEACE (Perú).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import joblib

MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest_model.joblib"

# Grilla de hiperparámetros para búsqueda exhaustiva
PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}

# Grilla reducida para desarrollo rápido (útil durante iteraciones)
PARAM_GRID_FAST: dict = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None],
}


def train_with_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fast_mode: bool = True,
    cv_folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple[RandomForestClassifier, GridSearchCV]:
    """
    Entrena un RandomForestClassifier usando GridSearchCV con StratifiedKFold.

    StratifiedKFold garantiza que cada fold mantenga la proporción de clases,
    esencial cuando hay desbalanceo entre éxito/fracaso.

    Args:
        X_train: Features de entrenamiento.
        y_train: Variable objetivo.
        fast_mode: Si True, usa PARAM_GRID_FAST (más rápido, menos exhaustivo).
        cv_folds: Número de folds para cross-validation.
        n_jobs: Número de núcleos CPU (-1 = todos).
        verbose: Nivel de detalle en consola.

    Returns:
        (best_model, grid_search_object)
    """
    param_grid = PARAM_GRID_FAST if fast_mode else PARAM_GRID

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",     # ROC AUC es la métrica objetivo principal del sistema
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,             # Re-entrena automáticamente con los mejores parámetros
        return_train_score=True,
    )

    print("Iniciando GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model: RandomForestClassifier = grid_search.best_estimator_

    print(f"\nMejores hiperparametros:\n{grid_search.best_params_}")
    print(f"Mejor F1 en CV: {grid_search.best_score_:.4f}")

    return best_model, grid_search


def run_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> dict[str, float]:
    """
    Ejecuta cross-validation sobre el modelo final con múltiples métricas.
    Esto proporciona estimaciones robustas de rendimiento para reportar en la tesis.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    metrics = {}

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        }
        print(f"CV {metric:>12}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {path}. Ejecute el pipeline de entrenamiento."
        )
    model = joblib.load(path)
    print(f"Modelo cargado desde: {path}")
    return model


def get_feature_importances(
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extrae y ordena las importancias de features (MDI - Mean Decrease Impurity).

    MDI mide cuanto reduce cada feature la impureza de Gini promediada sobre todos
    los arboles. Variables con mayor MDI son mas determinantes para la prediccion.
    Funciona tanto con RandomForestClassifier como con CalibratedClassifierCV.
    """
    # Extraer el RF base si el modelo fue calibrado
    if hasattr(model, "calibrated_classifiers_"):
        base_rf = model.calibrated_classifiers_[0].estimator
    else:
        base_rf = model

    importances = base_rf.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in base_rf.estimators_], axis=0
    )

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "std": std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\nTop {top_n} features más importantes:")
    print(df_importance.head(top_n).to_string(index=False))

    return df_importance
