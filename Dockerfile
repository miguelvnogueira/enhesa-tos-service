FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_REQUESTS_TIMEOUT=600 \
    PIP_TIMEOUT=600 \
    PIP_DEFAULT_TIMEOUT=600

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:/root/.local/bin:/usr/local/bin:$PATH"

COPY pyproject.toml poetry.lock* ./

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config installer.max-workers 2 && \
    poetry install --only main --no-root -vvv

COPY ./src /app/src
COPY ./models /app/models

EXPOSE 8000


CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]