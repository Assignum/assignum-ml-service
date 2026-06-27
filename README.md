# Assignum ML Service

Sistema de Machine Learning para la prediccion y recomendacion de asignacion de tareas colaborativas en entornos academicos universitarios. Desarrollado como parte de la tesis de Ingenieria de Software.

---

## Descripcion

El modelo predice la probabilidad de que un estudiante complete exitosamente una tarea colaborativa, considerando su perfil de habilidades tecnicas, competencias colaborativas, experiencia previa y disponibilidad horaria.

El sistema expone una API REST consumida por una aplicacion movil (Flutter) para sugerir automaticamente que estudiante deberia ser asignado a cada tarea dentro de un equipo.

---

## Arquitectura del proyecto

```
assignum-ml-service/
├── src/
│   ├── data_generation.py   # Generacion de dataset sintetico (1300 muestras)
│   ├── preprocessing.py     # Limpieza, feature engineering y encoding
│   ├── training.py          # Entrenamiento con GridSearchCV
│   ├── evaluation.py        # Metricas y graficas (escala de grises, 300 DPI)
│   ├── recommendation.py    # Ranking de candidatos por probabilidad de exito
│   └── retraining.py        # Reentrenamiento con datos reales
├── api/
│   ├── main.py              # Aplicacion FastAPI con todos los endpoints
│   ├── schemas.py           # Schemas Pydantic v2 (request/response)
│   └── predictor.py         # Singleton del modelo con lru_cache
├── tests/
│   └── test_api.py
├── main.py                  # CLI: entrena el modelo y corre los casos demo
├── demo_casos.py            # 5 casos de prueba del sistema de recomendacion
├── run_api.py               # Script para levantar la API
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

Ejecuta el pipeline completo: generacion de datos, preprocesamiento, entrenamiento, evaluacion y guardado del modelo. Al finalizar corre automaticamente los casos de prueba.

Opciones disponibles:

```bash
python main.py --fast           # GridSearch reducido (mas rapido, util para pruebas)
python main.py --samples 2000   # Cambiar el numero de muestras sinteticas
python main.py --demo           # Solo corre los casos demo (requiere modelo entrenado)
```

### 2. Levantar la API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentacion interactiva disponible en: `http://localhost:8000/docs`

---

## Metricas del modelo

Entrenado con 1300 muestras sinteticas y validado con StratifiedKFold (cv=5):

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
| GET | `/info` | Estado del modelo y encoders |
| POST | `/predict` | Prediccion individual (estudiante + tarea) |
| POST | `/recommend` | Ranking de candidatos para una tarea |
| POST | `/data/append` | Agregar registros de tareas completadas |
| POST | `/data/survey` | Incorporar encuestas reales de estudiantes |
| POST | `/model/retrain` | Disparar reentrenamiento en background |

### POST /predict

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

### POST /recommend

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
    {
      "name": "Luis",
      "backend_skill": 5,
      "available_hours_per_week": 30,
      "last_role": "Backend",
      "last_role_performance": 5,
      "peer_evaluation": 5
    },
    {
      "name": "Carlos",
      "backend_skill": 4,
      "available_hours_per_week": 12,
      "last_role": "Database",
      "last_role_performance": 3,
      "peer_evaluation": 3
    }
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
| `last_role` | enum | Ultimo rol en proyecto grupal |
| `last_role_performance` | 1-5 | Desempeno en ese ultimo rol |
| `peer_evaluation` | 1-5 | Evaluacion promedio de companeros |

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

El modelo calcula automaticamente estas features antes de predecir:

| Feature | Descripcion |
|---|---|
| `relevant_skill` | Habilidad del estudiante especifica para el `task_type` |
| `skill_complexity_gap` | `relevant_skill` - complejidad x 0.8 |
| `collab_avg` | Promedio de teamwork + communication + organization |
| `avail_hours_ratio` | `available_hours_per_week` / `estimated_hours` (max 5) |
| `role_matches_task` | 1 si `last_role` == `task_type`, 0 si no |
| `secondary_score` | Score compuesto: experiencia x0.30 + disponibilidad x0.25 + peer_eval x0.25 + desempeno x0.10 + rol x0.10 |

---

## Estrategia de datos

| Fase | Registros | Fuente |
|---|---|---|
| Fase 1 (actual) | 1300 | Dataset sintetico con reglas de dominio academico |
| Fase 2 | +300 | Encuestas reales con estudiantes (`POST /data/survey`) |
| Fase 3 | Continuo | Datos reales capturados por la app Flutter (`POST /data/append`) |

---

## Integracion con Flutter

URL base segun entorno:

| Entorno | URL |
|---|---|
| Emulador Android | `http://10.0.2.2:8000` |
| Dispositivo fisico (misma red) | `http://<IP-local-PC>:8000` |
| Produccion | `https://tu-dominio.com` |

Codigo base:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const String baseUrl = 'http://10.0.2.2:8000';

Future<Map<String, dynamic>> recommend({
  required Map<String, dynamic> task,
  required List<Map<String, dynamic>> students,
}) async {
  final response = await http.post(
    Uri.parse('$baseUrl/recommend'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'task': task, 'students': students}),
  );
  if (response.statusCode == 200) return jsonDecode(response.body);
  throw Exception('Error ${response.statusCode}: ${response.body}');
}
```

`pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
```

`AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

---

## Flujo de uso en la app

```
[Registro alumno]   -> captura skills (1-5) y datos de experiencia
        |
[Lider crea tarea]  -> define task_type, complexity, priority, estimated_hours
        |
[POST /recommend]   -> envia tarea + perfiles de los miembros del equipo
        |
[Muestra ranking]   -> recommended_student destacado, resto como lista ordenada
        |
[Tarea completada]  -> POST /data/append con success=1 o success=0
```

`success = 1` cuando: completada + a tiempo + lider_evaluacion >= 3 + peer_evaluacion >= 3
