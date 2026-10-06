# Predicción de velocidad de viento a corto plazo (H = 1–12 h)

Código del Trabajo Fin de Máster. Compara modelos estadísticos (SARIMA), de *machine learning*
y de *deep learning* para predecir la velocidad del viento horaria con horizontes de 1 a 12 horas,
usando como exógena la predicción numérica HARMONIE-AROME, y despliega el mejor modelo como servicio.

La etiqueta [`v1.0`](https://github.com/alvarochuecosd-spec/tfm-prediccion-viento/tree/v1.0)
corresponde a la versión del código descrita en la memoria (Anexo C).

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
| 8 | [8_MLOps_tracking.ipynb](8_MLOps_tracking.ipynb) | Registro de experimentos y del modelo en MLflow |

## Despliegue

- [app.py](app.py): API de inferencia (FastAPI) que sirve el modelo de [modelo_export/](modelo_export/).
- [streamlit_dashboard.py](streamlit_dashboard.py): panel de decisión *Human-in-the-loop*, que pide
  las predicciones a la API.
- [evidently_monitor.py](evidently_monitor.py): informe de deriva de datos (se genera en local; no se incluye en el repositorio).
  Compara las peticiones recibidas por la API con el conjunto de entrenamiento del modelo desplegado.
  La API guarda cada petición en un CSV si se define la variable de entorno `REGISTRO_PREDICCIONES`:

  ```bash
  REGISTRO_PREDICCIONES=registro/predicciones_recientes.csv uvicorn app:app --port 8000
  python evidently_monitor.py --actual registro/predicciones_recientes.csv --salida informes/
  ```
- [Dockerfile](Dockerfile), [Dockerfile.dashboard](Dockerfile.dashboard) y [docker-compose.yml](docker-compose.yml):

```bash
docker compose up --build
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-entrenamiento.txt   # notebooks (ejecutados con Python 3.14)
# o solo: pip install -r requirements.txt       # servicio de inferencia
```

## Datos

Los datos en crudo (observaciones de la estación y salidas de HARMONIE-AROME), los conjuntos
intermedios y las predicciones (`*.parquet`, que contienen las observaciones) **no se incluyen**
en el repositorio, ya que se han utilizado con permiso de sus propietarios. Están versionados con
DVC (ficheros `*.dvc`, ver [dvc_setup.md](dvc_setup.md)). Sí se incluyen las métricas (`*.csv`),
las configuraciones de los mejores modelos (`*.json`), las figuras que generan los notebooks
([figuras/](figuras/)) y las figuras empleadas en la memoria ([imgs/](imgs/)).

Sin los datos, el servicio que arranca completo es la API (`docker compose up --build api`): el
panel necesita `dataset_modelado.csv` y `harmonie_crudo.csv` para reconstruir las variables desde
el histórico, y la interfaz de MLflow necesita `mlflow.db`.