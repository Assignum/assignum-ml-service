# Assignum ML Service

Sistema de Machine Learning para prediccion y asignacion inteligente de tareas colaborativas en grupos academicos universitarios.

> Desarrollado como parte de la tesis de Ingenieria de Software.

---

## Que hace este sistema

El sistema tiene dos funciones principales:

**1. Recomendar** quien deberia hacer una tarea especifica — dado el perfil de cada estudiante y las caracteristicas de la tarea, el modelo predice la probabilidad de exito de cada candidato y retorna un ranking.

**2. Distribuir** un sprint completo entre todos los integrantes del grupo — dado un conjunto de tareas y estudiantes, el algoritmo asigna cada tarea considerando tanto la aptitud del estudiante como la equidad en la carga horaria.

Ambas funcionalidades se exponen como API REST consumible desde una app movil Flutter.

---

## Arquitectura

```
assignum-ml-service/
├── src/
│   ├── data_generation.py   # Genera dataset sintetico (1300 muestras con reglas de dominio)
│   ├── preprocessing.py     # Limpieza, feature engineering y label encoding
│   ├── training.py          # RandomForest + GridSearchCV (scoring: roc_auc)
│   ├── evaluation.py        # Metricas y graficas academicas (escala de grises, 300 DPI)
│   ├── recommendation.py    # Ranking de candidatos para una tarea individual
│   ├── assignment.py        # Distribucion equitativa de multiples tareas en un grupo
│   └── retraining.py        # Reentrenamiento con datos reales acumulados
├── api/
│   ├── main.py              # FastAPI app — todos los endpoints
│   ├── schemas.py           # Schemas Pydantic v2 (validacion de entrada y salida)
│   └── predictor.py         # Singleton del modelo con lru_cache
├── tests/
│   └── test_api.py
├── main.py                  # CLI: entrena el modelo y ejecuta los casos demo
├── demo_casos.py            # 5 casos de recomendacion + 5 casos de asignacion grupal
├── run_api.py               # Shortcut para levantar la API
└── requirements.txt
```

---

## Requisitos

- Python 3.11+
- pip

---

## Instalacion

```bash
git clone https://github.com/Assignum/assignum-ml-service.git
cd assignum-ml-service
pip install -r requirements.txt
```

---

## Uso

### 1. Entrenar el modelo

```bash
python main.py
```

Ejecuta el pipeline completo: generacion de datos sinteticos, preprocesamiento, entrenamiento con GridSearchCV, evaluacion y guardado del modelo. Al finalizar corre automaticamente los 10 casos de prueba (5 de recomendacion + 5 de asignacion grupal).

```bash
python main.py --demo           # Solo ejecuta los casos demo (requiere modelo ya entrenado)
python main.py --fast           # GridSearch reducido, util para pruebas rapidas
python main.py --samples 2000   # Genera dataset con N muestras sinteticas
```

### 2. Levantar la API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentacion interactiva en: `http://localhost:8000/docs`

---

## Metricas del modelo

Entrenado con 1300 muestras sinteticas, validado con StratifiedKFold (cv=5):

| Metrica  | Valor  |
|----------|--------|
| Accuracy | ~0.970 |
| F1 Score | ~0.970 |
| ROC AUC  | ~0.989 |

---

## Endpoints de la API

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Estado del modelo y encoders en disco |
| POST | `/predict` | Prediccion individual para un par (estudiante, tarea) |
| POST | `/recommend` | Ranking de candidatos para una tarea |
| POST | `/assign` | Distribucion de un sprint completo entre el grupo |
| POST | `/data/append` | Registrar tarea completada (alimenta reentrenamiento) |
| POST | `/data/survey` | Incorporar encuestas reales de estudiantes (Fase 2) |
| POST | `/model/retrain` | Disparar reentrenamiento en background |

---

### POST /predict

Predice la probabilidad de exito de un estudiante para una tarea especifica.

**Request:**
```json
{
  "backend_skill": 4,
  "frontend_skill": 2,
  "database_skill": 3,
  "testing_skill": 3,
  "documentation_skill": 2,
  "git_github_skill": 4,
  "agile_methodologies_skill": 3,
  "teamwork_skill": 4,
  "communication_skill": 3,
  "leadership_skill": 2,
  "organization_skill": 3,
  "projects_completed": 5,
  "available_hours_per_week": 20,
  "last_role": "Backend",
  "last_role_performance": 4,
  "peer_evaluation": 4,
  "task_type": "Backend",
  "task_complexity": "Medium",
  "priority": "High",
  "estimated_hours": 15
}
```

**Response:**
```json
{
  "success_probability": 0.8423,
  "prediction": "Exitoso",
  "confidence_level": "Alta"
}
```

---

### POST /recommend

Dado una tarea y una lista de candidatos, retorna el ranking ordenado por probabilidad de exito.

**Request:**
```json
{
  "task": {
    "task_type": "Backend",
    "task_complexity": "High",
    "priority": "High",
    "estimated_hours": 25
  },
  "students": [
    { "name": "Luis",   "backend_skill": 5, "available_hours_per_week": 30, "last_role": "Backend",   "last_role_performance": 5, "peer_evaluation": 5 },
    { "name": "Carlos", "backend_skill": 4, "available_hours_per_week": 12, "last_role": "Database",  "last_role_performance": 3, "peer_evaluation": 3 }
  ]
}
```

**Response:**
```json
{
  "recommended_student": "Luis",
  "success_probability": 0.9012,
  "ranking": [
    { "name": "Luis",   "success_probability": 0.9012, "prediction": "Exitoso", "rank": 1 },
    { "name": "Carlos", "success_probability": 0.7634, "prediction": "Exitoso", "rank": 2 }
  ]
}
```

---

### POST /assign

Distribuye multiples tareas de un sprint entre los integrantes del grupo, balanceando aptitud ML y equidad de carga horaria.

**Algoritmo:** cada tarea se asigna al estudiante con mayor `score = probabilidad_ML x (1 - carga_relativa_acumulada)`. Las tareas mas complejas se procesan primero para que los mas capaces las reciban antes de quedarse sin capacidad.

**Request:**
```json
{
  "tasks": [
    { "task_id": "T1", "task_name": "Auth con JWT",      "task_type": "Backend",       "task_complexity": "High",   "priority": "High",   "estimated_hours": 18 },
    { "task_id": "T2", "task_name": "UI de registro",    "task_type": "Frontend",      "task_complexity": "Medium", "priority": "High",   "estimated_hours": 10 },
    { "task_id": "T3", "task_name": "Esquema de BD",     "task_type": "Database",      "task_complexity": "Medium", "priority": "Medium", "estimated_hours": 8  },
    { "task_id": "T4", "task_name": "Sprint planning",   "task_type": "Management",    "task_complexity": "Low",    "priority": "High",   "estimated_hours": 3  }
  ],
  "students": [
    { "name": "Luis",    "backend_skill": 5,  "available_hours_per_week": 25, "last_role": "Backend",    "last_role_performance": 5, "peer_evaluation": 5 },
    { "name": "Ana",     "frontend_skill": 5, "available_hours_per_week": 20, "last_role": "Frontend",   "last_role_performance": 4, "peer_evaluation": 4 },
    { "name": "Carlos",  "database_skill": 5, "available_hours_per_week": 20, "last_role": "Database",   "last_role_performance": 4, "peer_evaluation": 3 },
    { "name": "Valeria", "leadership_skill": 5, "available_hours_per_week": 15, "last_role": "Management", "last_role_performance": 5, "peer_evaluation": 5 }
  ]
}
```

**Response:**
```json
{
  "assignments": [
    { "task_name": "Auth con JWT",    "assigned_to": "Luis",    "success_probability": 0.94, "estimated_hours": 18 },
    { "task_name": "UI de registro",  "assigned_to": "Ana",     "success_probability": 0.98, "estimated_hours": 10 },
    { "task_name": "Esquema de BD",   "assigned_to": "Carlos",  "success_probability": 0.98, "estimated_hours": 8  },
    { "task_name": "Sprint planning", "assigned_to": "Valeria", "success_probability": 0.92, "estimated_hours": 3  }
  ],
  "workload": [
    { "student_name": "Luis",    "total_hours": 18, "available_hours": 25, "utilization_pct": 72.0 },
    { "student_name": "Ana",     "total_hours": 10, "available_hours": 20, "utilization_pct": 50.0 },
    { "student_name": "Carlos",  "total_hours": 8,  "available_hours": 20, "utilization_pct": 40.0 },
    { "student_name": "Valeria", "total_hours": 3,  "available_hours": 15, "utilization_pct": 20.0 }
  ]
}
```

---

## Variables del modelo

### Perfil del estudiante

| Variable | Rango | Descripcion |
|---|---|---|
| `backend_skill` | 1-5 | Desarrollo backend |
| `frontend_skill` | 1-5 | Desarrollo frontend |
| `database_skill` | 1-5 | Bases de datos |
| `testing_skill` | 1-5 | Testing / QA |
| `documentation_skill` | 1-5 | Documentacion tecnica |
| `git_github_skill` | 1-5 | Control de versiones |
| `agile_methodologies_skill` | 1-5 | Metodologias agiles |
| `teamwork_skill` | 1-5 | Trabajo en equipo |
| `communication_skill` | 1-5 | Comunicacion |
| `leadership_skill` | 1-5 | Liderazgo |
| `organization_skill` | 1-5 | Organizacion y planificacion |
| `projects_completed` | 0-50 | Proyectos academicos completados |
| `available_hours_per_week` | 1-80 | Horas disponibles por semana |
| `last_role` | enum | Ultimo rol desempenado en un proyecto grupal |
| `last_role_performance` | 1-5 | Desempeno en ese ultimo rol |
| `peer_evaluation` | 1-5 | Evaluacion promedio recibida de companeros |

**Valores validos para `last_role`:**
`Backend` | `Frontend` | `Testing` | `Database` | `Documentation` | `Management`

**Escala de habilidades:**

| Valor | Descripcion |
|---|---|
| 1 | Sin conocimiento |
| 2 | Basico |
| 3 | Intermedio |
| 4 | Avanzado |
| 5 | Experto |

### Descripcion de la tarea

| Variable | Valores validos |
|---|---|
| `task_type` | `Backend`, `Frontend`, `Testing`, `Database`, `Documentation`, `Management` |
| `task_complexity` | `Low`, `Medium`, `High` |
| `priority` | `Low`, `Medium`, `High` |
| `estimated_hours` | 1 - 200 |

---

## Feature Engineering

El modelo calcula automaticamente estas features a partir de los datos crudos:

| Feature | Formula / Descripcion |
|---|---|
| `relevant_skill` | Habilidad del estudiante correspondiente al `task_type` |
| `skill_complexity_gap` | `relevant_skill` - complejidad x 0.8 |
| `collab_avg` | Promedio de teamwork + communication + organization |
| `avail_hours_ratio` | `available_hours_per_week` / `estimated_hours` (max 5) |
| `role_matches_task` | 1 si `last_role` == `task_type`, 0 si no |
| `secondary_score` | experiencia x0.30 + disponibilidad x0.25 + peer_eval x0.25 + desempeno x0.10 + rol x0.10 |

El mapeo `task_type -> habilidad relevante` es:
`Backend->backend_skill` | `Frontend->frontend_skill` | `Testing->testing_skill` | `Database->database_skill` | `Documentation->documentation_skill` | `Management->leadership_skill`

---

## Estrategia de datos

| Fase | Registros | Fuente |
|---|---|---|
| Fase 1 (actual) | 1300 | Dataset sintetico generado con reglas de dominio academico |
| Fase 2 | +300 | Encuestas reales con estudiantes universitarios (`POST /data/survey`) |
| Fase 3 | Continuo | Datos reales capturados por la app Flutter (`POST /data/append`) |

---

## Casos de prueba

`demo_casos.py` incluye 10 casos de prueba que se pueden correr con `python main.py --demo`:

**Recomendacion individual (5 casos):**
- Mismo skill, diferente disponibilidad
- Mismo skill, diferente experiencia y rol
- Tarea Management: liderazgo vs skill tecnica
- Tres clones con distinta disponibilidad
- Zona media — factores secundarios deciden

**Asignacion grupal (5 casos):**
- Sprint con roles claros por especialidad
- Misma habilidad — carga 100% equitativa
- Experto + juniors — tareas complejas al experto, simples a juniors
- Grupo sobrecargado — fallback sin crash
- Sprint realista con 4 integrantes y 6 tareas mixtas

---

## Integracion con Flutter

URL base segun entorno:

| Entorno | URL |
|---|---|
| Emulador Android | `http://10.0.2.2:8000` |
| Dispositivo fisico (misma red) | `http://<IP-local-PC>:8000` |
| Produccion | `https://tu-dominio.com` |

`pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
```

`AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

Ejemplo de llamada al endpoint de asignacion:
```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const String baseUrl = 'http://10.0.2.2:8000';

Future<Map<String, dynamic>> assignTasks({
  required List<Map<String, dynamic>> tasks,
  required List<Map<String, dynamic>> students,
}) async {
  final response = await http.post(
    Uri.parse('$baseUrl/assign'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'tasks': tasks, 'students': students}),
  );
  if (response.statusCode == 200) return jsonDecode(response.body);
  throw Exception('Error ${response.statusCode}: ${response.body}');
}
```

---

## Flujo de uso en la app

```
[Registro alumno]      -> captura skills (1-5) y datos de experiencia
        |
[Lider crea sprint]    -> registra todas las tareas del sprint
        |
[POST /assign]         -> envia tareas + perfiles de todos los integrantes
        |
[Muestra asignacion]   -> cada tarea con su asignado y el resumen de carga por persona
        |
[Tarea completada]     -> POST /data/append con success=1 o success=0
```

`success = 1` cuando: completada + a tiempo + lider_evaluacion >= 3 + peer_evaluacion >= 3
