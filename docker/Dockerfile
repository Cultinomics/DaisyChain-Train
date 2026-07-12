# DaisyChain node image (CPU torch keeps it lean; GPU nodes use a real machine).
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy psutil
WORKDIR /app
COPY daisychain/ ./daisychain/
COPY examples/ ./examples/
COPY config/ ./config/
COPY docker/node_entrypoint.sh ./node_entrypoint.sh
RUN chmod +x ./node_entrypoint.sh
ENV GLOO_SOCKET_IFNAME=eth0
ENV USE_LIBUV=0
ENV PYTHONUNBUFFERED=1
ENV DAISY_STATUS_FILE=/app/status.json
CMD ["./node_entrypoint.sh"]
