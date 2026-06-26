# Use a slim Python 3.11 base image to match your pyproject.toml
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install curl, then download and run the official poetry installer script
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install official Poetry to a dedicated global directory
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy only dependency management files first to maximize Docker layer caching
COPY pyproject.toml poetry.lock* ./

# Install project dependencies excluding development groups
RUN poetry install --only main --no-root

# Copy application source code and precomputed model artifacts
COPY ./src /app/src
COPY ./models /app/models

# Expose the API port
EXPOSE 8000

# Run the Uvicorn application server
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]