# ============================================================================
# Dockerfile — Sistema de Riesgo Crediticio con RBM y RAG Educativo
# ============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Dependencias del sistema:
#   - libgomp1     : requerido por xgboost / lightgbm
#   - curl         : usado por el HEALTHCHECK
#   - build-essential : por si alguna wheel necesita compilarse
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar torch en su versión CPU ANTES que el resto.
# Evita descargar ~2.5 GB de wheels con CUDA que esta app no usa.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencias de Python (capa cacheada: solo se reinstala si cambia requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código de la aplicación
COPY . .

# Entrypoint: genera .streamlit/secrets.toml a partir de la variable GROQ_API_KEY
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
