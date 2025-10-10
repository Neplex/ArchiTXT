ARG PYTHON_VERSION=3.12

FROM python:$PYTHON_VERSION AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    PIP_CACHE_DIR=/root/.cache/pip

ARG POETRY_VERSION=1.8.2
RUN --mount=type=cache,id=pip,target=$PIP_CACHE_DIR \
    pip install --upgrade setuptools wheel poetry==$POETRY_VERSION poetry-plugin-export

# Build dependecies
COPY pyproject.toml ./
RUN --mount=type=cache,id=pip,target=$PIP_CACHE_DIR \
    poetry export --no-cache --format=requirements.txt --all-extras --without=dev --output=requirements.txt \
    && pip wheel --prefer-binary --requirement requirements.txt --wheel-dir /wheels \
    && rm requirements.txt

# Build app
COPY architxt ./architxt
RUN poetry build --no-cache --format=wheel --output=/wheels


FROM python:$PYTHON_VERSION-slim AS runtime

WORKDIR /app

# Install app and dependencies from wheelhouse
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --no-deps /wheels/*.whl

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONDEVMODE=0 \
    PYTHONHASHSEED=random

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["python", "-m", "architxt"]
CMD ["ui", "--server.port=8080", "--server.address=0.0.0.0"]
