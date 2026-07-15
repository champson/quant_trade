FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

ENV QT_CONFIG=/app/configs/default.yaml
EXPOSE 8501
CMD ["qt", "dashboard", "--port", "8501"]

