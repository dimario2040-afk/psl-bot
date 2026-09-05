FROM python:3.11-slim

WORKDIR /app

# Только нужные файлы (контекст чистят .dockerignore).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Hugging Face Spaces прокидывает HTTP на 7860 — задай PORT=7860 в Secrets.
CMD ["python", "bot.py"]
