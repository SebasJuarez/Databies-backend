# databies-backend

Backend de cálculo de vigas para la página de Diagramas de Databies. Flask + [Structo](https://pypi.org/project/databies/) (`pip install databies`).

## Rutas

| Ruta | Qué hace |
|---|---|
| `POST /compute` | Resuelve la viga (también responde en `/structo/compute`) |
| `GET /health` | Estado y versión de Structo; la página lo llama al abrir Diagramas para despertar el servidor |

La viga se resuelve dos veces: `service` (cargas nominales, para deflexión y rotación) y `factored` (mayoradas, para V, M y reacciones de diseño). Las dos van en `data.bases`. Detalle de la petición, unidades y signos al inicio de `app.py`.

## Correr en local

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python app.py        # http://localhost:8080
.venv/bin/pytest               # pruebas
```

## Render

- Python ≥ 3.11 (Structo lo pide). El repo trae `.python-version` con 3.12; si el servicio tiene la variable `PYTHON_VERSION`, esa manda.
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`
- Opcional: `CORS_ORIGINS` con los dominios permitidos separados por coma (por defecto `*`).
