# Reddit-research MCP server — HTTP transport for container/NAS deployment.
# Uses the official uv image (uv preinstalled) for fast, reproducible installs.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer) using the locked resolution.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mcp

ENV PATH="/app/.venv/bin:$PATH"

# Build identity, printed in the startup log. Pass GIT_SHA (and optionally
# BUILD_DATE) as build args; the date falls back to the time of the build.
# Kept after the dependency install so changing them doesn't bust that cache.
ARG GIT_SHA=unknown
ARG BUILD_DATE=
RUN printf 'commit=%s built=%s\n' "$GIT_SHA" "${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}" \
    > /app/BUILD_INFO

# HTTP transport bound to all interfaces inside the container.
ENV REDDIT_RESEARCH_MCP_TRANSPORT=streamable-http \
    REDDIT_RESEARCH_MCP_HOST=0.0.0.0 \
    REDDIT_RESEARCH_MCP_PORT=8000 \
    REDDIT_RESEARCH_CACHE=/data/reddit-research.sqlite

# Cache lives on a mounted volume so it survives restarts.
VOLUME ["/data"]
EXPOSE 8000

# python-only health check (no curl in the slim image); hits the unauthenticated /healthz.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["reddit-research-mcp"]
