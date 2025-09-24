FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

# Set environment variables for better async performance
ENV PYTHONUNBUFFERED=1
ENV ASYNCIO_DEBUG=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="$PYTHONPATH:/app/src"

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "wal_fact_checker.a2a:a2a_app", "--host", "0.0.0.0", "--port", "8080"]