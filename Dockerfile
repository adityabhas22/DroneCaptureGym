FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DRONECAPTUREOPS_MAX_CONCURRENT_ENVS=64

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY dronecaptureops ./dronecaptureops
COPY server ./server
COPY inference.py ./inference.py

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
