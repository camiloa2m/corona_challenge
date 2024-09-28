FROM python:3.9-slim

RUN useradd -ms /bin/bash appuser

WORKDIR /app_classifier

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && pip list

USER appuser

COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser model/ model/

# Exponer el puerto que usará FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
