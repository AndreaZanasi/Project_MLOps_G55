FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE


ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Last layer so if it changes it skips the above steps
COPY src/ src/
COPY configs/ configs/

ENTRYPOINT ["uv", "run", "src/proj/train.py"]