FROM python:3.11-slim

WORKDIR /app

COPY photo_processor.py /app/
COPY config.yaml /app/

# Install dependencies
RUN pip install --no-cache-dir opencv-python-headless mediapipe pillow rembg watchdog pyyaml numpy

# Set environment variables (optional defaults)
ENV CONFIG_FILE=/app/config.yaml

CMD ["python", "photo_processor.py"]
