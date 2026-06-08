FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPSTREAM_BASE_URL=http://127.0.0.1:8000
ENV UPSTREAM_MODEL=vl-model
ENV REQUEST_TIMEOUT=120
ENV DROP_UNSUPPORTED_PARAMS=true
ENV STRIP_REASONING=true

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 18000

CMD ["python", "-m", "uvicorn", "mineru_adapter.api:app", "--host", "0.0.0.0", "--port", "18000"]
