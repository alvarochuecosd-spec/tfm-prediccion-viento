"""
Dashboard Human-in-the-loop — predicción de viento, Tarifa.

Consume la API REST (app.py), no reimplementa la carga del modelo: así el
dashboard nunca puede desincronizarse de lo que sirve producción.

    streamlit run streamlit_dashboard.py

Requiere la API en marcha (uvicorn o Docker):
    curl http://localhost:8000/salud   ->   {"estado": "ok", ...}
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

RUTA = Path(__file__).parent
API_URL_DEFECTO = os.environ.get("API_URL", "http://localhost:8000")
LOG_DECISIONES = RUTA / "decisiones_hitl.csv"
DATASET = RUTA / "dataset_modelado.csv"
CRUDO = RUTA / "harmonie_crudo.csv"
OBJ = "obs_vel"


# --------------------------------------------------------------------------
# Lógica de negocio, separada de la interfaz para poder probarla sin Streamlit.
# --------------------------------------------------------------------------
def clasificar(pred: float, umbral: float, banda: float) -> str:
    """
    Tres estados en lugar de una decisión binaria:

      ACTIVAR         -> incluso restando la banda de error se supera el umbral.
      NO_ACTIVAR      -> incluso sumándola se queda por debajo.
      REVISION_HUMANA -> el umbral cae dentro del margen de error del modelo:
                         la decisión automática no es fiable, decide un operador.
    """
    if pred - banda >= umbral:
        return "ACTIVAR"
    if pred + banda < umbral:
        return "NO_ACTIVAR"
    return "REVISION_HUMANA"


COLOR = {"ACTIVAR": "#c62828", "NO_ACTIVAR": "#2e7d32", "REVISION_HUMANA": "#ef6c00"}
TEXTO = {
    "ACTIVAR": "ACTIVAR PROTOCOLO DE SEGURIDAD",
    "NO_ACTIVAR": "SIN RIESGO — no se requiere acción",
    "REVISION_HUMANA": "REVISIÓN HUMANA REQUERIDA",
}


@st.cache_data(ttl=600)
def cargar_contexto(H: int, L: int) -> pd.DataFrame | None:
    """
    Histórico para autocompletar las features de una hora concreta.

    Las variables de HARMONIE-AROME se toman de la pasada disponible en el instante de
    decisión t-H (lead >= H + L, Ecuación 3.1 de la memoria), igual que construir_dataset()
    del notebook 7. Las columnas H_* de dataset_modelado.csv no sirven aquí: proceden de la
    pasada más reciente con lead >= L, sin tener en cuenta el horizonte.
    """
    if not (DATASET.exists() and CRUDO.exists()):
        return None
    crudo = pd.read_csv(CRUDO, usecols=["valida", "pasada", "lead", "WSPE", "WSPN"],
                        parse_dates=["valida", "pasada"])
    apto = crudo[crudo["lead"] >= H + L].sort_values(["valida", "pasada"])
    oper = apto.groupby("valida").last()
    direccion = (np.degrees(np.arctan2(-oper["WSPE"], -oper["WSPN"])) + 360) % 360
    harm = pd.DataFrame({
        "H_vel": np.hypot(oper["WSPE"], oper["WSPN"]),
        "H_lead": oper["lead"],
        "H_dir_sin": np.sin(np.radians(direccion)),
        "H_dir_cos": np.cos(np.radians(direccion)),
    })
    harm.index.name = "ts"

    base = pd.read_csv(DATASET, index_col="ts", parse_dates=["ts"])
    base = base.drop(columns=[c for c in base.columns if c.startswith("H_")], errors="ignore")
    return base.join(harm, how="inner")


def autocompletar(ts_str: str, features: list[str], H: int, W: int, L: int = 4) -> dict | None:
    """
    Reconstruye las features de una hora del histórico con la misma lógica que
    construir_dataset() de los notebooks 4 y 7: lags que terminan en t-H, pronóstico de
    HARMONIE-AROME de la pasada disponible en el instante de decisión y agregados de la
    ventana.
    """
    d = cargar_contexto(H, L)
    if d is None:
        return None
    try:
        ts = pd.Timestamp(ts_str)
    except ValueError:
        return None
    if ts not in d.index:
        return None

    fila = d.loc[ts]
    tramo = fila["tramo"] if "tramo" in d.columns else None
    g = d[d.tramo == tramo].sort_index() if tramo is not None else d.sort_index()
    if ts not in g.index:
        return None
    pos = g.index.get_loc(ts)

    vals = {}
    ventana = []
    for k in range(H, H + W):
        j = pos - k
        if j < 0:
            return None
        v = float(g[OBJ].iloc[j])
        vals[f"lag{k}"] = v
        ventana.append(v)

    for c in ("hora_sin", "hora_cos", "dia_sin", "dia_cos",
              "H_vel", "H_dir_sin", "H_dir_cos", "H_lead"):
        if c in g.columns:
            vals[c] = float(fila[c])

    arr = np.array(ventana)
    vals["lag_media"] = float(arr.mean())
    vals["lag_std"] = float(arr.std(ddof=1))   # ddof=1, como pandas .std() en el notebook 7
    vals["lag_tend"] = float(arr[0] - arr[-1])

    faltan = [f for f in features if f not in vals]
    if faltan:
        return None
    return {f: vals[f] for f in features}


def registrar(fila: dict) -> None:
    """Traza cada decisión: sin registro no hay auditoría ni reentrenamiento futuro."""
    df = pd.DataFrame([fila])
    df.to_csv(LOG_DECISIONES, mode="a", header=not LOG_DECISIONES.exists(), index=False)


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Viento Tarifa — HITL", page_icon="🌬", layout="wide")
    st.title("Predicción de viento — Tarifa")
    st.caption("Corrección estadística de HARMONIE-AROME · sistema con supervisión humana")

    with st.sidebar:
        st.header("Configuración")
        api_url = st.text_input("URL de la API", API_URL_DEFECTO)

        try:
            salud = requests.get(f"{api_url}/salud", timeout=3).json()
        except requests.exceptions.RequestException as e:
            st.error(f"API no disponible: {type(e).__name__}")
            st.info("Arranca la API:\\n\\n`uvicorn app:app --port 8000`")
            st.stop()

        if salud.get("estado") != "ok":
            st.error(f"La API responde pero el modelo no está cargado: {salud}")
            st.stop()

        st.success("API operativa")
        st.metric("Horizonte", f"{salud['horizonte_h']} h")

        cfg = {}
        if (RUTA / "modelo_produccion.json").exists():
            cfg = json.loads((RUTA / "modelo_produccion.json").read_text())
            st.metric("MAE del modelo", f"{cfg.get('mae_test', float('nan')):.3f} m/s")

        umbral = st.number_input("Umbral de activación (m/s)", 0.0, 40.0,
                                 float(cfg.get("umbral_activacion_m_s", 11.5)), 0.5)
        banda = st.slider(
            "Banda de incertidumbre (m/s)", 0.0, 5.0,
            float(round(cfg.get("mae_test", 1.5), 2)), 0.1,
            help="Margen alrededor del umbral dentro del cual la decisión se escala "
                 "a un operador. Por defecto, el MAE del modelo.",
        )

    features = cfg.get("features", [])
    H = int(salud.get("horizonte_h", 12))
    W = int(cfg.get("ventana", 12))
    L = int(cfg.get("latencia_harmonie_h", 4))

    tab_pred, tab_hist = st.tabs(["Predicción", "Histórico de decisiones"])

    # ---------------------------------------------------------------- pestaña 1
    with tab_pred:
        col_in, col_out = st.columns([1, 1])

        with col_in:
            st.subheader("1. Datos de entrada")
            modo = st.radio("Origen", ["Hora del histórico", "Entrada manual"],
                            horizontal=True, label_visibility="collapsed")

            entrada = None
            ts_obj = None

            if modo == "Hora del histórico":
                d = cargar_contexto(H, L)
                if d is None:
                    st.warning("No se encuentran dataset_modelado.csv y harmonie_crudo.csv para autocompletar.")
                else:
                    ts_sel = st.selectbox(
                        "Hora objetivo",
                        options=[str(x) for x in d.index[-500:][::-1]],
                        help="Se reconstruyen las features de esa hora igual que en entrenamiento.",
                    )
                    entrada = autocompletar(ts_sel, features, H, W, L)
                    ts_obj = ts_sel
                    if entrada is None:
                        st.warning("No hay suficiente historia previa para esa hora.")
                    else:
                        with st.expander("Ver features reconstruidas"):
                            st.dataframe(
                                pd.Series(entrada, name="valor").to_frame().round(4),
                                width="stretch",
                            )
            else:
                st.caption("Valores principales; el resto se completa con la mediana histórica.")
                d = cargar_contexto(H, L)
                base_vals = {}
                if d is not None:
                    for f in features:
                        if f in d.columns:
                            base_vals[f] = float(d[f].median())
                        elif f.startswith("lag"):
                            base_vals[f] = float(d[OBJ].median())
                        else:
                            base_vals[f] = 0.0
                else:
                    base_vals = {f: 0.0 for f in features}

                c1, c2 = st.columns(2)
                with c1:
                    h_vel = st.number_input("HARMONIE — velocidad (m/s)", 0.0, 40.0, 8.0, 0.5)
                    h_dir = st.slider("HARMONIE — dirección (°)", 0, 359, 90)
                with c2:
                    vel_reciente = st.number_input("Velocidad observada reciente (m/s)",
                                                   0.0, 40.0, 7.0, 0.5)
                    hora = st.slider("Hora del día (UTC)", 0, 23, 12)

                base_vals.update({
                    "H_vel": h_vel,
                    "H_dir_sin": float(np.sin(np.radians(h_dir))),
                    "H_dir_cos": float(np.cos(np.radians(h_dir))),
                    "hora_sin": float(np.sin(2 * np.pi * hora / 24)),
                    "hora_cos": float(np.cos(2 * np.pi * hora / 24)),
                })
                for f in features:
                    if f.startswith("lag") and f[3:].isdigit():
                        base_vals[f] = vel_reciente
                lags = [base_vals[f] for f in features if f.startswith("lag") and f[3:].isdigit()]
                if lags:
                    base_vals["lag_media"] = float(np.mean(lags))
                    base_vals["lag_std"] = float(np.std(lags))
                    base_vals["lag_tend"] = 0.0
                entrada = {f: float(base_vals.get(f, 0.0)) for f in features}
                ts_obj = datetime.now(timezone.utc).isoformat()

        with col_out:
            st.subheader("2. Resultado")
            if entrada is None:
                st.info("Selecciona o introduce los datos de entrada.")
            elif st.button("Predecir", type="primary", width="stretch"):
                try:
                    r = requests.post(f"{api_url}/predecir",
                                      json={"features": entrada, "ts_objetivo": ts_obj},
                                      timeout=15)
                    r.raise_for_status()
                    res = r.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error al llamar a la API: {e}")
                    st.stop()

                pred = res["velocidad_predicha_m_s"]
                estado = clasificar(pred, umbral, banda)
                st.session_state["ultimo"] = {"pred": pred, "estado": estado,
                                              "ts_obj": ts_obj, "res": res}

            if "ultimo" in st.session_state:
                u = st.session_state["ultimo"]
                pred, estado = u["pred"], u["estado"]

                st.markdown(
                    f"<div style='background:{COLOR[estado]};color:white;padding:1.2rem;"
                    f"border-radius:.5rem;text-align:center;font-size:1.15rem;"
                    f"font-weight:600'>{TEXTO[estado]}</div>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicción", f"{pred:.2f} m/s")
                c2.metric("Umbral", f"{umbral:.1f} m/s", delta=f"{pred - umbral:+.2f}")
                c3.metric("Intervalo", f"±{banda:.2f}")
                st.caption(f"Rango considerado: {max(pred - banda, 0):.2f} – {pred + banda:.2f} m/s")

                if estado == "REVISION_HUMANA":
                    st.warning(
                        "El umbral cae dentro del margen de error del modelo. "
                        "La decisión automática no es fiable: debe decidir un operador."
                    )

                st.subheader("3. Decisión del operador")
                with st.form("decision"):
                    accion = st.radio("Acción tomada",
                                      ["Activar protocolo", "No activar", "Aplazar"],
                                      horizontal=True)
                    nota = st.text_input("Justificación (opcional)")
                    if st.form_submit_button("Registrar decisión", width="stretch"):
                        registrar({
                            "ts_decision": datetime.now(timezone.utc).isoformat(),
                            "ts_objetivo": u["ts_obj"],
                            "prediccion_m_s": pred,
                            "umbral_m_s": umbral,
                            "banda_m_s": banda,
                            "estado_sistema": u["estado"],
                            "decision_operador": accion,
                            "coincide_con_sistema": (
                                (accion == "Activar protocolo" and u["estado"] == "ACTIVAR")
                                or (accion == "No activar" and u["estado"] == "NO_ACTIVAR")
                            ),
                            "justificacion": nota,
                        })
                        st.success("Decisión registrada.")

    # ---------------------------------------------------------------- pestaña 2
    with tab_hist:
        st.subheader("Histórico de decisiones")
        if not LOG_DECISIONES.exists():
            st.info("Todavía no se ha registrado ninguna decisión.")
        else:
            log = pd.read_csv(LOG_DECISIONES)
            c1, c2, c3 = st.columns(3)
            c1.metric("Decisiones", len(log))
            if "estado_sistema" in log.columns:
                escaladas = (log.estado_sistema == "REVISION_HUMANA").sum()
                c2.metric("Escaladas a revisión", f"{escaladas} ({100*escaladas/len(log):.0f}%)")
            if "coincide_con_sistema" in log.columns:
                ac = log.coincide_con_sistema.sum()
                c3.metric("Operador de acuerdo", f"{ac} ({100*ac/len(log):.0f}%)")
            st.dataframe(log.iloc[::-1], width="stretch")
            st.download_button("Descargar CSV", log.to_csv(index=False),
                               "decisiones_hitl.csv", "text/csv")
            st.caption(
                "Las discrepancias entre operador y sistema son la materia prima para "
                "reentrenar: señalan dónde el modelo no captura el criterio experto."
            )


if __name__ == "__main__":
    main()
