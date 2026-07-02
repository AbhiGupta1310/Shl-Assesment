FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files needed for installation
COPY pyproject.toml README.md ./
COPY app/ app/

# Install dependencies
RUN uv pip install --system -e .

# Install Playwright browser
RUN uv pip install --system playwright && playwright install chromium --with-deps

EXPOSE 8000

# Bind port dynamically from $PORT variable or default to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
