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
        disp = RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax,
            name="Random Forest",
        )
        disp.line_.set_color("black")
        disp.line_.set_linewidth(2)
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


# Rangos de horas disponibles por semana usados en la revisión de equidad
HOURS_BINS = [0, 10, 20, float("inf")]
HOURS_LABELS = ["<=10 h", "11-20 h", ">20 h"]


def subgroup_fairness_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoders: dict | None = None,
    threshold: float = 0.5,
    max_gap: float = 0.10,
    min_samples: int = 30,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Revisión de equidad: calcula el F1 por subgrupo y lo compara con el F1 global.

    Subgrupos:
      - available_hours_per_week en rangos (<=10, 11-20, >20): evita que el modelo
        perjudique a quien trabaja o tiene menos tiempo libre.
      - last_role: evita que un rol previo concreto quede sistemáticamente relegado.

    Un subgrupo queda marcado (flagged) si su F1 está más de `max_gap` por debajo del
    global. Con menos de `min_samples` registros no se evalúa: el F1 sería poco fiable.
    El reporte se guarda como CSV junto a feature_importance.png.
    """
    y_true = pd.Series(y_test).reset_index(drop=True)
    X = X_test.reset_index(drop=True)
    y_pred = pd.Series((model.predict_proba(X)[:, 1] >= threshold).astype(int))
    global_f1 = f1_score(y_true, y_pred, zero_division=0)

    hours_group = pd.cut(X["available_hours_per_week"], bins=HOURS_BINS, labels=HOURS_LABELS)
    role_group = X["last_role"]
    if encoders and "last_role" in encoders:
        role_group = pd.Series(encoders["last_role"].inverse_transform(role_group.astype(int)))

    # Primera fila: el F1 global, que es la referencia contra la que se compara cada subgrupo
    rows = [{"group": "global", "subgroup": "todos", "n": len(y_true),
             "f1": round(global_f1, 4), "gap_vs_global": 0.0, "status": "reference"}]
    for group_name, groups in (("available_hours_per_week", hours_group), ("last_role", role_group)):
        for subgroup in sorted(groups.dropna().unique(), key=str):
            mask = (groups == subgroup).to_numpy()
            n = int(mask.sum())
            if n < min_samples:
                rows.append({"group": group_name, "subgroup": str(subgroup), "n": n,
                             "f1": np.nan, "gap_vs_global": np.nan, "status": "insufficient_data"})
                continue
            f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
            gap = global_f1 - f1
            rows.append({"group": group_name, "subgroup": str(subgroup), "n": n,
                         "f1": round(f1, 4), "gap_vs_global": round(gap, 4),
                         "status": "flagged" if gap > max_gap else "ok"})

    report = pd.DataFrame(rows)
    report.attrs["global_f1"] = global_f1

    print("\n" + "=" * 50)
    print(f"FAIRNESS REVIEW - F1 POR SUBGRUPO (global F1 = {global_f1:.4f})")
    print("=" * 50)
    print(report.to_string(index=False))
    flagged = report[report["status"] == "flagged"]
    if len(flagged):
        print(f"\nATENCION: {len(flagged)} subgrupo(s) con F1 mas de {max_gap:.0%} por debajo del global.")
    else:
        print(f"\nNingun subgrupo queda mas de {max_gap:.0%} por debajo del F1 global.")

    path = save_path or (PLOTS_DIR / "fairness_report.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"Reporte de equidad guardado en: {path}")
    return report


def full_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    encoders: dict | None = None,
) -> dict:
    """Ejecuta evaluación completa, genera todos los gráficos y la revisión de equidad."""
    metrics = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(importance_df)
    subgroup_fairness_report(model, X_test, y_test, encoders=encoders)
    return metrics
