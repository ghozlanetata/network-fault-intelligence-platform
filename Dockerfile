FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    KERAS_HOME=/home/appuser/.keras \
    NFI_MODEL_DIR=/app/models \
    NFI_DATABASE_PATH=/app/data/nfi.sqlite3

WORKDIR /app

# TensorFlow's Linux runtime uses OpenMP; keep the OS runtime layer minimal.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid 10001 --create-home \
        --home-dir /home/appuser --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/models /home/appuser/.keras \
    && chown -R 10001:10001 /app/data /app/models /home/appuser

COPY pyproject.toml requirements-ml.txt ./
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations

# requirements-ml.txt delegates to the pyproject's single [ml] dependency set.
RUN python -m pip install --no-cache-dir -r requirements-ml.txt

USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
