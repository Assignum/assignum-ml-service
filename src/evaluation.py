"""
evaluation.py
Evaluación completa del modelo: métricas, matriz de confusión y visualizaciones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

PLOTS_DIR = Path(__file__).parent.parent / "data" / "processed"


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula todas las métricas de evaluación sobre el conjunto de test.

    El umbral (threshold) puede ajustarse para optimizar precision/recall
    según el contexto: en asignación de tareas, un FN (no recomendar a alguien
    competente) es menos costoso que un FP (recomendar a alguien no preparado).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n" + "=" * 50)
    print("EVALUATION METRICS - TEST SET")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:>12}: {value:.4f}")
    print()
    print("Detailed report:")
    print(classification_report(y_test, y_pred, target_names=["Unsuccessful", "Successful"]))

    return metrics


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greys",
            xticklabels=["Unsuccessful", "Successful"],
            yticklabels=["Unsuccessful", "Successful"],
            linewidths=0.5,
            linecolor="black",
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("Confusion Matrix - Random Forest Classifier", fontsize=12, fontweight="bold")
        plt.tight_layout()

        path = save_path or (PLOTS_DIR / "confusion_matrix.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    print(f"Matriz de confusion guardada en: {path}")


def plot_roc_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> None:
    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax,
            name="Random Forest",
            color="black",
            lw=2,
        )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random classifier")
        ax.set_title("ROC Curve - Random Forest Classifier", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")
        plt.tight_layout()

        path = save_path or (PLOTS_DIR / "roc_curve.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    print(f"Curva ROC guardada en: {path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    top = importance_df.head(top_n).copy()

    # Paleta de grises degradados: la barra más importante es negra,
    # la menos importante es gris claro — facilita lectura en papel impreso.
    n = len(top)
    grays = [str(round(0.1 + 0.6 * i / max(n - 1, 1), 2)) for i in range(n)]
    colors = list(reversed(grays))  # más oscuro = más importante (arriba en barh)

    with plt.style.context("grayscale"):
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            top["feature"][::-1],
            top["importance"][::-1],
            xerr=top["std"][::-1],
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"ecolor": "black", "capsize": 3, "linewidth": 1},
        )
        ax.set_xlabel("Feature Importance (MDI - Mean Decrease Impurity)", fontsize=11)
        ax.set_title(
            f"Top {top_n} Most Important Features - Random Forest Classifier",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.4, color="gray")
        ax.set_xlim(left=0)
        plt.tight_layout()

        path = save_path or (PLOTS_DIR / "feature_importance.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
    print(f"Feature importance guardada en: {path}")


def full_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
) -> dict:
    """Ejecuta evaluación completa y genera todos los gráficos."""
    metrics = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(importance_df)
    return metrics
