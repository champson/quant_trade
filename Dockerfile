FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir .

ENV QT_CONFIG=/app/configs/default.yaml
EXPOSE 8501
CMD ["qt", "dashboard", "--port", "8501"]
