## Que cambia

<!-- Describe brevemente que hace este PR -->

## Por que

<!-- Motivo del cambio: bug, nueva feature, mejora de metricas, etc. -->

## Tipo de cambio

- [ ] Nueva funcionalidad
- [ ] Correccion de bug
- [ ] Mejora del modelo o metricas
- [ ] Documentacion
- [ ] Refactor

## Como probarlo

```bash
# Ejemplo de como reproducir o verificar el cambio
python main.py --demo
```

## Checklist

- [ ] El modelo sigue entrenando sin errores (`python main.py`)
- [ ] Los casos demo producen resultados coherentes (`python main.py --demo`)
- [ ] La API levanta sin errores (`uvicorn api.main:app --reload`)
- [ ] No se suben archivos `.joblib` ni `.csv` generados
