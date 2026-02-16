# Stage 1: Build the Next.js static files
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./
RUN npm ci

# Copy all files
COPY . .

# Build argument for Clerk public key
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ENV NEXT_TELEMETRY_DISABLED=1

# Build the Next.js app (creates 'out' directory with static files)
RUN npm run build

# Stage 2: Create the final Python container (slim runtime)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python + dependencies
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip "setuptools<70" wheel \
    && pip install --no-cache-dir --no-build-isolation -r requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy the FastAPI server package
COPY api ./api

# Persist logs across container restarts
RUN mkdir -p /app/api/logs
VOLUME ["/app/api/logs"]

# Copy the Next.js static export from builder stage
COPY --from=frontend-builder /app/out ./static

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Expose port 8000 (FastAPI will serve everything)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
