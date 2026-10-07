FROM python:3.11-slim

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_CACHE=/app/models \
    DATA_DIR=/data \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the model at BUILD time so the container starts fast
RUN python -c "from fastembed import TextEmbedding; import os; m=TextEmbedding(model_name=os.environ['MODEL_NAME'], cache_dir=os.environ['MODEL_CACHE']); list(m.embed(['warm up']))"

COPY app ./app
RUN mkdir -p /data

EXPOSE 8000
# Render injects $PORT; one worker keeps memory low
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
