"""
Monitorización de deriva del modelo — predicción de viento, Tarifa.

Compara la distribución de las features y del error del modelo entre un periodo de
referencia (los datos con los que se entrenó/evaluó, p.ej. el conjunto de test del
notebook 4b) y un periodo "actual" (las predicciones más recientes en producción).

Pensado para ejecutarse periódicamente (cron, tarea programada), no de forma interactiva.
No es un notebook porque un proceso de monitorización continua no tiene "una ejecución
con una salida"; tiene ejecuciones repetidas que se acumulan en un histórico de alertas.

Uso:
    python evidently_monitor.py --referencia dataset_modelado.csv \
        --actual predicciones_recientes.csv --salida informes/

Programación con cron (ejemplo, cada día a las 06:00):
    0 6 * * * cd /ruta/al/proyecto && python evidently_monitor.py \
        --referencia dataset_modelado.csv --actual /var/log/viento/ultimas_24h.csv \
        --salida informes/ >> logs/monitor.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("monitor-deriva")

# ============================================================================
# Columnas a vigilar. Deben coincidir con las features del modelo en producción
# (ver modelo_produccion.json) más el objetivo, para poder comparar también el
# error si el fichero "actual" incluye la observación real.
# ============================================================================
COLUMNAS_NUMERICAS = [
    "H_vel", "H_dir_sin", "H_dir_cos", "H_lead",
    "lag_media", "lag_std", "lag_tend",
    "obs_vel",
]

UMBRAL_ALERTA_SHARE = 0.3   # si >= 30% de las columnas derivan, se dispara alerta


def cargar(ruta: Path, columnas: list[str]) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    disponibles = [c for c in columnas if c in df.columns]
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        logger.warning("Columnas ausentes en %s (se omiten): %s", ruta, faltan)
    return df[disponibles].dropna()


def generar_informe(ref: pd.DataFrame, actual: pd.DataFrame) -> tuple[Report, dict]:
    columnas_comunes = [c for c in ref.columns if c in actual.columns]
    dd = DataDefinition(numerical_columns=columnas_comunes)

    ds_ref = Dataset.from_pandas(ref[columnas_comunes], data_definition=dd)
    ds_actual = Dataset.from_pandas(actual[columnas_comunes], data_definition=dd)

    report = Report([DataDriftPreset()])
    resultado = report.run(reference_data=ds_ref, current_data=ds_actual)
    return resultado, resultado.dict()


def interpretar(resultado_dict: dict) -> dict:
    """Extrae un resumen legible: qué columnas derivan y si se supera el umbral de alerta."""
    resumen = {"columnas_con_deriva": [], "share_global": None, "alerta": False}

    for m in resultado_dict["metrics"]:
        nombre = m["metric_name"]
        if nombre.startswith("DriftedColumnsCount"):
            resumen["share_global"] = m["value"]["share"]
            resumen["n_columnas_derivadas"] = int(m["value"]["count"])
        elif nombre.startswith("ValueDrift"):
            columna = m["config"]["column"]
            p_valor = m["value"]
            if p_valor < m["config"]["threshold"]:
                resumen["columnas_con_deriva"].append({"columna": columna, "p_valor": p_valor})

    if resumen["share_global"] is not None:
        resumen["alerta"] = resumen["share_global"] >= UMBRAL_ALERTA_SHARE
    return resumen


def notificar(resumen: dict, destino_email: str | None) -> None:
    mensaje = (
        f"ALERTA DE DERIVA — {resumen['n_columnas_derivadas']} columnas afectadas "
        f"({resumen['share_global']*100:.0f}%)\n\n"
        + "\n".join(f"  - {c['columna']}: p={c['p_valor']:.2e}"
                    for c in resumen["columnas_con_deriva"])
    )
    logger.warning(mensaje)

    if not destino_email:
        return
    try:
        msg = MIMEText(mensaje)
        msg["Subject"] = "Deriva detectada — modelo de predicción de viento"
        msg["From"] = "monitor@viento-tarifa.local"
        msg["To"] = destino_email
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
        logger.info("Notificación enviada a %s", destino_email)
    except Exception as e:
        # Un fallo de envío de email NUNCA debe tirar el proceso de monitorización:
        # la alerta ya quedó registrada en el log y en el informe HTML/JSON.
        logger.error("No se pudo enviar el email de alerta: %s", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencia", type=Path, required=True,
                    help="CSV con los datos de referencia (p.ej. test del notebook 4b)")
    ap.add_argument("--actual", type=Path, required=True,
                    help="CSV con los datos recientes a comparar")
    ap.add_argument("--salida", type=Path, default=Path("informes_deriva"),
                    help="Carpeta donde guardar el informe HTML/JSON")
    ap.add_argument("--email", type=str, default=None,
                    help="Dirección a la que notificar si hay alerta (opcional)")
    args = ap.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    logger.info("Cargando referencia: %s", args.referencia)
    ref = cargar(args.referencia, COLUMNAS_NUMERICAS)
    logger.info("Cargando actual: %s", args.actual)
    actual = cargar(args.actual, COLUMNAS_NUMERICAS)

    if len(actual) < 30:
        logger.warning(
            "Solo %d filas en el periodo actual: el test de deriva no es fiable con "
            "muestras tan pequeñas. Se genera igualmente, pero interpretar con cautela.",
            len(actual),
        )

    resultado, resultado_dict = generar_informe(ref, actual)
    resumen = interpretar(resultado_dict)

    ruta_html = args.salida / f"deriva_{ts}.html"
    ruta_json = args.salida / f"deriva_{ts}.json"
    resultado.save_html(str(ruta_html))
    with open(ruta_json, "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Informe guardado: %s", ruta_html)
    logger.info("Resumen: %d/%d columnas con deriva (%.0f%%)",
               resumen.get("n_columnas_derivadas", 0), len(ref.columns),
               (resumen["share_global"] or 0) * 100)

    if resumen["alerta"]:
        notificar(resumen, args.email)
    else:
        logger.info("Sin alerta: la deriva está por debajo del umbral (%.0f%%)",
                   UMBRAL_ALERTA_SHARE * 100)


if __name__ == "__main__":
    main()
