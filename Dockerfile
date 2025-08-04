FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание кеш-директорий с правильными правами
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface /app/.cache/datasets /app/.cache/matplotlib && \
    chmod -R 777 /app/.cache/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY pages/ ./pages/
COPY images/ ./images/
COPY app.py ./
COPY .streamlit/ ./.streamlit/

# Исправленные переменные окружения
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/datasets
ENV MPLCONFIGDIR=/app/.cache/matplotlib
# Убираем устаревшую TRANSFORMERS_CACHE - она теперь использует HF_HOME

# Повторная установка прав перед запуском
RUN chmod -R 777 /app/.cache/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
