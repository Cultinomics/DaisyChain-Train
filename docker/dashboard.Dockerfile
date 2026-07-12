# DaisyChain dashboard image — pure stdlib, no torch. Runs server.py standalone
# (its scanner import falls back to a flat import, so no torch package init).
FROM python:3.11-slim
WORKDIR /app
COPY daisychain/dashboard/server.py daisychain/dashboard/scanner.py ./
COPY config/ ./config/
ENV PYTHONUNBUFFERED=1
ENV DAISY_NODES_FILE=config/nodes.example.json
EXPOSE 8080
CMD ["python", "server.py"]
