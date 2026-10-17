# syntax=docker/dockerfile:1
#
# Runtime image for TMAL. PyTorch is installed from the cu128 wheel index
# and ships its own CUDA 12.8 runtime libraries (see the nvidia-*-cu12 deps
# in uv.lock), so this image only needs the NVIDIA driver + Container
# Toolkit on the host (`docker run --gpus all ...`) -- no CUDA toolkit base
# image required.
#
# `apex` in pyproject.toml resolves to an unrelated PyPI package (not
# NVIDIA/apex); src/solver*.py already falls back to a no-op AMP shim when
# `from apex import amp` fails, so no source build of real apex is needed.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Runtime libs needed by opencv-python (non-headless) and friends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Install dependencies first so this layer is cached across non-src edits.
# `src/` has to be present too: uv needs the setuptools "src" package layout
# on disk to validate the local project against uv.lock, even with
# --no-install-project.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Now bring in the rest of the project and install it too.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# `data/` and `ckpts/` are gitignored and expected to be bind-mounted at
# runtime, e.g.:
#   docker run --gpus all -it \
#     -v $(pwd)/data:/app/data -v $(pwd)/ckpts:/app/ckpts tmal bash
CMD ["bash"]
