ARG PYTHON_VERSION=3.13

FROM python:$PYTHON_VERSION AS builder

WORKDIR /build

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0

RUN apt-get update  \
    && apt-get install --no-install-recommends -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

ARG POETRY_VERSION=2.2.1
RUN pip install --no-cache-dir --upgrade \
    pip==25.3 \
    setuptools==80.9.0 \
    wheel==0.45.1 \
    poetry==$POETRY_VERSION \
    poetry-plugin-export==1.9.0

# Build dependencies
COPY pyproject.toml ./
RUN poetry --no-cache export --format=requirements.txt --all-extras --output=requirements.txt \
    && pip wheel --no-cache-dir --prefer-binary --requirement requirements.txt --wheel-dir /build/wheels \
    && rm requirements.txt

FROM python:$PYTHON_VERSION-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --create-home --uid 10001 appuser

# Install dependencies from wheelhouse
RUN --mount=type=bind,from=builder,source=/build/wheels,target=/build/wheels \
    pip install --no-cache-dir --break-system-packages --no-index --no-deps /build/wheels/*.whl

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
