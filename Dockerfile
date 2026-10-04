# Imagen de inferencia: solo lo necesario para SERVIR el modelo.
# El entrenamiento (mlflow, optuna, lightgbm, torch) queda fuera a propósito.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El modelo y su contrato viajan DENTRO de la imagen: la versión de la imagen
# identifica unívocamente qué modelo se está sirviendo.
COPY app.py modelo_produccion.json ./
COPY modelo_export/booster.json ./modelo_export/booster.json

# Usuario sin privilegios: limita el impacto de una hipotética ejecución remota.
RUN useradd --create-home --shell /bin/bash servicio && chown -R servicio:servicio /app
USER servicio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
