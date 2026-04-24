FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install only server deps (not torch/trl — too large for HF Space)
RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic openai python-dotenv

COPY src/ ./src/
COPY tasks/ ./tasks/
COPY server/ ./server/
COPY inference.py .
COPY train.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .

ENV PYTHONPATH=/app
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "server.app"]
