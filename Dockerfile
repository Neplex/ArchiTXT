ARG PYTHON_VERSION=3.12

FROM python:$PYTHON_VERSION AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/root/.cache/poetry \
    PIP_CACHE_DIR=/root/.cache/pip

RUN apt-get update  \
    && apt-get install --no-install-recommends -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

ARG POETRY_VERSION=2.2.1
RUN --mount=type=cache,id=pip,target=$PIP_CACHE_DIR \
    pip install --upgrade pip setuptools wheel poetry==$POETRY_VERSION poetry-plugin-export

# Build dependencies
COPY pyproject.toml ./
RUN --mount=type=cache,id=poetry,target=$POETRY_CACHE_DIR \
    --mount=type=cache,id=pip,target=$PIP_CACHE_DIR \
    poetry export --format=requirements.txt --all-extras --output=requirements.txt \
    && pip wheel --prefer-binary --requirement requirements.txt --wheel-dir /wheels \
    && rm requirements.txt

FROM python:$PYTHON_VERSION-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --create-home --uid 10001 appuser

# Install dependencies from wheelhouse
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --break-system-packages --no-index --no-deps /wheels/*.whl

COPY --chown=appuser:appuser architxt ./architxt
USER appuser

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONDEVMODE=0 \
    PYTHONHASHSEED=random

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["python", "-m", "architxt"]
CMD ["ui", "--server.port=8080", "--server.address=0.0.0.0"]
