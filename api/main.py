"""
API REST con FastAPI para el modelo de predicción de éxito en tareas colaborativas.

Endpoints:
    GET  /               → Health check
    GET  /info           → Información del modelo
    POST /predict        → Predicción individual
    POST /recommend      → Ranking de candidatos para una tarea
    POST /assign         → Distribucion equitativa de multiples tareas en un grupo
    POST /data/append    → Agregar nuevos registros al dataset histórico
    POST /model/retrain  → Disparar reentrenamiento
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.schemas import (
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    CandidateResponse,
    AssignRequest,
    AssignResponse,
    TaskAssignmentResponse,
    WorkloadSummaryResponse,
    NewTaskRecord,
    AppendDataResponse,
    RetrainResponse,
)
from api.predictor import predict_single, recommend_for_task, reload_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Assignum ML API",
    description=(
        "API para predicción de éxito en asignación de tareas colaborativas "
        "universitarias — Tesis de Ingeniería de Software, ULIMA / UPC."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Info ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Assignum ML API", "version": "1.0.0"}


@app.get("/info", tags=["Health"])
def model_info():
    """Retorna información básica del modelo cargado."""
    from src.training import MODEL_PATH
    from src.preprocessing import ENCODERS_PATH
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "encoders_path": str(ENCODERS_PATH),
        "encoders_exist": ENCODERS_PATH.exists(),
    }


# ─── Predicción individual ────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(request: PredictRequest):
    """
    Predice la probabilidad de que un estudiante complete exitosamente
    una tarea específica.

    **Retorna:**
    - `success_probability`: Probabilidad entre 0 y 1.
    - `prediction`: "Exitoso" o "No Exitoso".
    - `confidence_level`: "Alta", "Media" o "Baja" según la certeza del modelo.
    """
    try:
        prob, prediction, confidence = predict_single(request)
        return PredictResponse(
            success_probability=round(prob, 4),
            prediction=prediction,
            confidence_level=confidence,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─── Recomendación ────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["Recomendación"])
def recommend(request: RecommendRequest):
    """
    Dado una tarea y una lista de estudiantes candidatos, retorna el ranking
    ordenado de mayor a menor probabilidad de éxito.

    El primer estudiante del ranking es la **recomendación** del sistema.
    """
    try:
        results = recommend_for_task(request.task, request.students)
        ranking = [
            CandidateResponse(
                name=r.name,
                success_probability=round(r.success_probability, 4),
                prediction=r.prediction,
                rank=idx + 1,
            )
            for idx, r in enumerate(results)
        ]
        best = results[0]
        return RecommendResponse(
            recommended_student=best.name,
            success_probability=round(best.success_probability, 4),
            ranking=ranking,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error en /recommend")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─── Asignacion grupal ────────────────────────────────────────────────────────

@app.post("/assign", response_model=AssignResponse, tags=["Asignacion"])
def assign(request: AssignRequest):
    """
    Distribuye multiples tareas de un sprint entre los integrantes de un grupo
    de forma equitativa, balanceando aptitud del estudiante y carga horaria.

    **Logica de asignacion:**
    - Las tareas mas complejas (mas horas) se asignan primero.
    - Cada tarea va al estudiante con mayor probabilidad de exito que
      todavia tenga capacidad horaria disponible.
    - Si el grupo esta sobrecargado, se asigna igualmente al mas apto.

    **Retorna:**
    - `assignments`: que tarea fue asignada a que estudiante.
    - `workload`: resumen de horas y porcentaje de uso por estudiante.
    """
    try:
        from src.assignment import assign_tasks, TaskItem, StudentItem
        from api.predictor import get_model, get_encoders

        tasks = [
            TaskItem(
                task_id=t.task_id,
                task_name=t.task_name,
                task_type=t.task_type,
                task_complexity=t.task_complexity,
                priority=t.priority,
                estimated_hours=t.estimated_hours,
            )
            for t in request.tasks
        ]
        students = [
            StudentItem(
                name=s.name,
                backend_skill=s.backend_skill,
                frontend_skill=s.frontend_skill,
                database_skill=s.database_skill,
                testing_skill=s.testing_skill,
                documentation_skill=s.documentation_skill,
                git_github_skill=s.git_github_skill,
                agile_methodologies_skill=s.agile_methodologies_skill,
                teamwork_skill=s.teamwork_skill,
                communication_skill=s.communication_skill,
                leadership_skill=s.leadership_skill,
                organization_skill=s.organization_skill,
                projects_completed=s.projects_completed,
                available_hours_per_week=s.available_hours_per_week,
                last_role=s.last_role,
                last_role_performance=s.last_role_performance,
                peer_evaluation=s.peer_evaluation,
            )
            for s in request.students
        ]

        model = get_model()
        encoders = get_encoders()
        assignments, workload = assign_tasks(tasks, students, model, encoders)

        return AssignResponse(
            assignments=[
                TaskAssignmentResponse(
                    task_id=a.task_id,
                    task_name=a.task_name,
                    assigned_to=a.assigned_to,
                    success_probability=a.success_probability,
                    estimated_hours=a.estimated_hours,
                    task_type=a.task_type,
                    task_complexity=a.task_complexity,
                    priority=a.priority,
                )
                for a in assignments
            ],
            workload=[
                WorkloadSummaryResponse(
                    student_name=w.student_name,
                    assigned_tasks=w.assigned_tasks,
                    total_hours=w.total_hours,
                    available_hours=w.available_hours,
                    utilization_pct=w.utilization_pct,
                )
                for w in workload
            ],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error en /assign")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─── Gestión de datos ─────────────────────────────────────────────────────────

@app.post("/data/survey", response_model=AppendDataResponse, tags=["Datos"])
def append_survey(records: list[NewTaskRecord]):
    """
    Incorpora registros reales recopilados de encuestas con estudiantes.

    Endpoint diseñado para la **Fase 2** de la tesis: alimentar el modelo
    con los ~300 registros reales recopilados en campo, complementando
    los 1200 registros sintéticos de la Fase 1.

    El campo `success` puede calcularse externamente (1/0) o derivarse
    de los criterios: tarea completada + a tiempo + evaluación positiva.
    """
    try:
        from src.retraining import append_survey_data
        data = [r.model_dump() for r in records]
        total = append_survey_data(data)
        return AppendDataResponse(
            message=f"{len(data)} registros de encuesta incorporados.",
            total_records=total,
        )
    except Exception as e:
        logger.exception("Error en /data/survey")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/append", response_model=AppendDataResponse, tags=["Datos"])
def append_data(records: list[NewTaskRecord]):
    """
    Agrega nuevos registros de tareas completadas al dataset histórico.
    Estos datos se usarán en el siguiente ciclo de reentrenamiento.
    """
    try:
        from src.retraining import append_new_data
        data = [r.model_dump() for r in records]
        total = append_new_data(data)
        return AppendDataResponse(
            message=f"{len(data)} registros agregados exitosamente.",
            total_records=total,
        )
    except Exception as e:
        logger.exception("Error en /data/append")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Reentrenamiento ──────────────────────────────────────────────────────────

@app.post("/model/retrain", response_model=RetrainResponse, tags=["Modelo"])
def retrain(background_tasks: BackgroundTasks, force: bool = False):
    """
    Dispara el reentrenamiento del modelo en segundo plano.

    - `force=false` (default): Solo reentrenará si hay >= 50 nuevos registros
      y si el nuevo modelo supera al actual en ROC AUC.
    - `force=true`: Fuerza el reentrenamiento y reemplazo inmediato.

    **Nota:** El proceso se ejecuta en background. Use GET /info para confirmar
    que el modelo fue actualizado.
    """
    def _retrain_task():
        from src.retraining import retrain_model
        updated = retrain_model(force=force)
        if updated:
            reload_model()

    background_tasks.add_task(_retrain_task)
    return RetrainResponse(
        message="Reentrenamiento iniciado en background.",
        model_updated=False,
    )
