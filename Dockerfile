FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY inference.py .
COPY openenv.yaml .

# Create a simple API endpoint for HF Space
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import json\n\
from flask import Flask, request, jsonify\n\
from src.customer_support_env import CustomerSupportEnv, Action\n\
\n\
app = Flask(__name__)\n\
\n\
@app.route("/reset", methods=["POST"])\n\
def reset():\n\
    """Reset endpoint for OpenEnv validation"""\n\
    data = request.get_json() or {}\n\
    task_level = data.get("task_level", "easy")\n\
    env = CustomerSupportEnv(task_level=task_level)\n\
    obs = env.reset()\n\
    return jsonify({\n\
        "status": "ok",\n\
        "observation": obs.model_dump()\n\
    })\n\
\n\
@app.route("/health", methods=["GET"])\n\
def health():\n\
    return jsonify({"status": "healthy"})\n\
\n\
if __name__ == "__main__":\n\
    port = int(os.getenv("PORT", 7860))\n\
    app.run(host="0.0.0.0", port=port)\n\
' > app.py

# Install Flask for API
RUN pip install flask

# Expose port
EXPOSE 7860

# Run the API server
CMD ["python", "app.py"]
