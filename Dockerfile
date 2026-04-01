ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION AS builder

WORKDIR /build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y build-essential git

ARG POETRY_VERSION=2.3.3
RUN pip install --no-cache-dir --upgrade \
    pip==25.3 \
    setuptools==80.9.0 \
    wheel==0.45.1 \
    poetry==$POETRY_VERSION \
    poetry-plugin-export==1.9.0

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0

# Build dependencies
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=cache,target=/root/.cache/pip \
    poetry --no-cache export --format=requirements.txt --all-extras --output=requirements.txt \
    && pip wheel --prefer-binary --requirement requirements.txt --wheel-dir /build/wheels \
    && rm requirements.txt

FROM python:$PYTHON_VERSION-slim AS runtime

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y curl

# Install dependencies from wheelhouse
RUN --mount=type=bind,from=builder,source=/build/wheels,target=/build/wheels \
    pip install --no-cache-dir --no-index --no-deps /build/wheels/*.whl

RUN useradd --system --create-home --uid 10001 appuser
RUN mkdir -p /app/data && chown -R appuser:appuser /app

COPY --chown=appuser:appuser architxt /app/architxt

USER appuser
WORKDIR /app
VOLUME /app/data

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONDEVMODE=0 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=sqlite:////app/data/mlflow.db

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "architxt"]
CMD ["ui", "--server.port=8080", "--server.address=0.0.0.0"]
