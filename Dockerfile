FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY tasks/ ./tasks/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 7860

# Run the OpenEnv server
CMD ["python", "-m", "server.app"]
