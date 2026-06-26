FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    # CRITICAL: Broadening global timeout boundaries for massive neural wheels
    POETRY_REQUESTS_TIMEOUT=600 \
    PIP_TIMEOUT=600 \
    PIP_DEFAULT_TIMEOUT=600

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install official Poetry to a dedicated global directory
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Append local and global package bin paths to the environment PATH variable
ENV PATH="$POETRY_HOME/bin:/root/.local/bin:/usr/local/bin:$PATH"

# Copy only dependency management files first to maximize Docker layer caching
COPY pyproject.toml poetry.lock* ./

# FIX: Expand installer configuration parameters to gracefully handle heavy streams
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config installer.max-workers 2 && \
    poetry install --only main --no-root -vvv

# Copy application source code and precomputed model artifacts
COPY ./src /app/src
COPY ./models /app/models

# Expose the API port
EXPOSE 8000


CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]