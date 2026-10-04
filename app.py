"""
API REST de predicción de viento — Tarifa.

Sirve el modelo XGBoost campeón (ver notebook 7) como un servicio HTTP.

Carga el modelo en formato NATIVO de XGBoost (Booster.save_model/load_model),
no a través del wrapper de scikit-learn ni del Model Registry de MLflow. Dos
problemas reales que esto evita:

  1. Portabilidad de rutas: con almacenamiento de artefactos en disco local,
     MLflow graba en su base de datos la ruta ABSOLUTA de la máquina donde se
     registró el modelo. Esa ruta no existe dentro de un contenedor Docker.
  2. Compatibilidad de versiones: el wrapper XGBRegressor de scikit-learn
     serializa también sus atributos internos, que cambian entre versiones
     mayores de xgboost/scikit-learn (falla con TypeError: `_estimator_type`
     undefined si el host que entrenó y el contenedor que sirve no coinciden
     exactamente). El formato nativo del Booster es estable entre versiones:
     es precisamente para lo que existe.

Como consecuencia, este servicio NO necesita tener `mlflow` instalado — solo
`xgboost`. Eso también acelera el build de la imagen Docker.

Ejecución local (sin Docker):
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Documentación interactiva una vez arrancado:
    http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("viento-api")

RUTA = Path(__file__).parent
CONFIG_PATH = RUTA / "modelo_produccion.json"

app = FastAPI(
    title="Predicción de viento — Tarifa",
    description=(
        "Corrección estadística de HARMONIE-AROME para activación de seguridad "
        "en paneles solares. Ver TFM para metodología completa."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Estado cargado una vez al arrancar el servicio, no en cada petición.
# ---------------------------------------------------------------------------
_estado: dict = {}


@app.on_event("startup")
def cargar_modelo() -> None:
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"No se encuentra {CONFIG_PATH}. Ejecuta el notebook 7 primero: "
            "genera modelo_produccion.json con la configuración de despliegue."
        )
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    ruta_booster = RUTA / config.get("modelo_booster_path", "modelo_export/booster.json")
    if not ruta_booster.exists():
        raise RuntimeError(
            f"No se encuentra {ruta_booster}. Genera el Booster nativo con el "
            "script de exportación (ver guía de despliegue) antes de arrancar la API."
        )

    logger.info("Cargando Booster nativo desde %s", ruta_booster)
    booster = xgb.Booster()
    booster.load_model(str(ruta_booster))

    _estado["config"] = config
    _estado["booster"] = booster
    _estado["features"] = config["features"]
    _estado["horizonte_h"] = config["horizonte_h"]
    _estado["umbral"] = config["umbral_activacion_m_s"]
    logger.info(
        "Modelo cargado: horizonte=%sh, %d features, umbral=%.1f m/s",
        _estado["horizonte_h"], len(_estado["features"]), _estado["umbral"],
    )


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------
class EntradaPrediccion(BaseModel):
    """
    Un registro con exactamente las features que el modelo espera (ver
    modelo_produccion.json -> 'features'). Se valida su presencia y tipo, pero NO
    se recalculan aquí: la construcción de lags/tendencia es responsabilidad del
    pipeline de ingesta (ETL), no de esta API.
    """
    features: dict[str, float] = Field(
        ...,
        description="Diccionario feature -> valor. Debe incluir todas las features del modelo.",
        examples=[{
            "lag12": 6.2, "lag13": 5.9, "lag14": 6.5, "lag15": 6.8, "lag16": 7.1, "lag17": 7.0,
            "hora_sin": 0.5, "hora_cos": 0.87, "dia_sin": 0.3, "dia_cos": 0.95,
            "H_vel": 8.5, "H_dir_sin": 0.98, "H_dir_cos": -0.17, "H_lead": 16,
            "lag_media": 6.58, "lag_std": 0.45, "lag_tend": -0.8,
        }],
    )
    ts_objetivo: str | None = Field(
        None, description="ISO-8601 de la hora objetivo, solo informativo en la respuesta."
    )


class SalidaPrediccion(BaseModel):
    velocidad_predicha_m_s: float
    activar_seguridad: bool
    umbral_m_s: float
    horizonte_h: int
    ts_objetivo: str | None
    ts_prediccion: str
    modelo: str
    alias: str


class Salud(BaseModel):
    estado: Literal["ok", "modelo_no_cargado"]
    modelo: str | None = None
    alias: str | None = None
    horizonte_h: int | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/salud", response_model=Salud)
def salud() -> Salud:
    if "booster" not in _estado:
        return Salud(estado="modelo_no_cargado")
    c = _estado["config"]
    return Salud(estado="ok", modelo=c["modelo_registrado"], alias=c["alias"],
                horizonte_h=c["horizonte_h"])


@app.post("/predecir", response_model=SalidaPrediccion)
def predecir(entrada: EntradaPrediccion) -> SalidaPrediccion:
    if "booster" not in _estado:
        raise HTTPException(503, "El modelo todavía no se ha cargado.")

    features = _estado["features"]
    faltan = [f for f in features if f not in entrada.features]
    if faltan:
        raise HTTPException(
            422,
            f"Faltan {len(faltan)} features requeridas por el modelo: {faltan}",
        )
    sobran = [f for f in entrada.features if f not in features]
    if sobran:
        logger.warning("Features ignoradas (no las usa el modelo): %s", sobran)

    fila = pd.DataFrame([{f: entrada.features[f] for f in features}])[features]

    try:
        dmatrix = xgb.DMatrix(fila, feature_names=features)
        pred = float(_estado["booster"].predict(dmatrix)[0])
    except Exception as e:
        logger.exception("Fallo al predecir")
        raise HTTPException(500, f"Error interno del modelo: {type(e).__name__}") from e

    # XGBoost en regresión estándar no impone no-negatividad: nada en el modelo
    # impide matemáticamente una predicción negativa, aunque sea físicamente
    # imposible para una velocidad de viento. Ocurre en zonas del espacio de
    # entrada poco representadas en el entrenamiento (extrapolación del árbol).
    # Se capa aquí, en el punto de servicio, y se deja constancia en el log
    # para poder auditar cuántas veces ocurre y si se concentra en algún
    # régimen concreto (p.ej. calmas de invierno).
    if pred < 0:
        logger.warning(
            "Predicción negativa capada a 0: %.3f m/s (posible extrapolación "
            "del modelo fuera de la distribución de entrenamiento)", pred,
        )
        pred = 0.0

    return SalidaPrediccion(
        velocidad_predicha_m_s=round(pred, 3),
        activar_seguridad=pred >= _estado["umbral"],
        umbral_m_s=_estado["umbral"],
        horizonte_h=_estado["horizonte_h"],
        ts_objetivo=entrada.ts_objetivo,
        ts_prediccion=datetime.now(timezone.utc).isoformat(),
        modelo=_estado["config"]["modelo_registrado"],
        alias=_estado["config"]["alias"],
    )


@app.get("/")
def raiz() -> dict:
    return {
        "servicio": "Predicción de viento — Tarifa",
        "endpoints": ["/salud", "/predecir (POST)", "/docs"],
    }
