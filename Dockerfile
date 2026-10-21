# Local API and pipeline. Runs as a non-root user.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chowk chowk
COPY data/labels data/labels
COPY data/junctions.json data/junctions.json
RUN useradd -m chowk && mkdir -p data/raw artifacts && chown -R chowk /app
USER chowk
EXPOSE 8000
CMD ["python", "-c", "import uvicorn; from chowk.api import create_app; uvicorn.run(create_app(), host='0.0.0.0', port=8000)"]
