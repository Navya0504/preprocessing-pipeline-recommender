
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/tmp/data

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY experiments/results.csv ./experiments/results.csv

RUN mkdir -p /tmp/data \
    && useradd --create-home appuser \
    && chown -R appuser:appuser /app /tmp/data

USER appuser

EXPOSE 7860

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
