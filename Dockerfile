FROM python:3.11-slim

WORKDIR /app

# Install Docker CLI (to communicate with host Docker via socket)
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8080

# Default command
CMD ["dock-api", "--host", "0.0.0.0", "--port", "8080"]
