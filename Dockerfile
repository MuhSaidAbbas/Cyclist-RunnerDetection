# =====================
# STAGE 1: BUILDER
# =====================
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install torch==2.1.0 torchvision==0.16.0 \
 && pip install -r requirements.txt

# =====================
# STAGE 2: RUNTIME
# =====================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
