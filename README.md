# Predicción de velocidad de viento a corto plazo (H = 1–12 h)

Código del Trabajo Fin de Máster. Compara modelos estadísticos (SARIMA), de *machine learning*
y de *deep learning* para predecir la velocidad del viento horaria con horizontes de 1 a 12 horas,
usando como exógena la predicción numérica HARMONIE-AROME, y despliega el mejor modelo como servicio.

## Notebooks

Se ejecutan en orden; cada uno lee las salidas del anterior.

| # | Notebook | Contenido |
|---|----------|-----------|
| 1 | [1_ETL.ipynb](1_ETL.ipynb) | De los ficheros en crudo al dataset horario |
| 2 | [2_EDA.ipynb](2_EDA.ipynb) | Análisis exploratorio |
| 3 | [3_SARIMA.ipynb](3_SARIMA.ipynb) | Baselines y SARIMA(X) con validación *walk-forward* |
| 4 | [4_ML.ipynb](4_ML.ipynb) | Modelos de *machine learning* |
| 5 | [5_DL.ipynb](5_DL.ipynb) | Modelos de *deep learning* |
| 6 | [6_ensemble_diebold_mariano.ipynb](6_ensemble_diebold_mariano.ipynb) | Ensembles y test de Diebold-Mariano |
| 7 | [7_despliegue.ipynb](7_despliegue.ipynb) | Exportación del modelo y servicio de inferencia |
| 8 | [8_MLOps_tracking.ipynb](8_MLOps_tracking.ipynb) | Seguimiento con MLflow y monitorización de deriva |

## Despliegue

- [app.py](app.py): API de inferencia (FastAPI) que sirve el modelo de [modelo_export/](modelo_export/).
- [streamlit_dashboard.py](streamlit_dashboard.py): cuadro de mando.
- [evidently_monitor.py](evidently_monitor.py): informe de deriva de datos (ejemplo en [informes/](informes/)).
- [Dockerfile](Dockerfile), [Dockerfile.dashboard](Dockerfile.dashboard) y [docker-compose.yml](docker-compose.yml):

```bash
docker compose up --build
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-entrenamiento.txt   # notebooks
# o solo: pip install -r requirements.txt       # servicio de inferencia
```

## Datos

Los datos en crudo (observaciones horarias y salidas de HARMONIE-AROME) y los datasets intermedios
**no se incluyen** en el repositorio. Están versionados con DVC (ficheros `*.dvc`, ver
[dvc_setup.md](dvc_setup.md)). Sí se incluyen los resultados necesarios para revisar el trabajo:
predicciones (`*.parquet`), métricas (`*.csv`), configuraciones de los mejores modelos (`*.json`)
y figuras ([figuras/](figuras/)).
