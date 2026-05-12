FROM python:3.12-slim

WORKDIR /app

# System deps for numpy/xgboost/scikit-learn wheels (most are prebuilt, but keep gcc for fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    STATE_DIR=/var/data

VOLUME ["/var/data"]

# Default: loop mode. Override with `python alert_system.py --once` for cron.
CMD ["python", "alert_system.py"]
