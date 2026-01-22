FROM nvcr.io/nvidia/pytorch:23.10-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE

COPY .dvc .dvc
COPY dvc.lock dvc.yaml ./

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src/ src/
COPY configs/ configs/

ENTRYPOINT ["/bin/bash", "-c", "uv run dvc pull && uv run src/proj/train.py"]
