# ================== Base Image ==================
FROM python:3.12-slim

# ================== Set Working Directory ==================
WORKDIR /app

# ================== Install System Dependencies with reliable mirror ==================
RUN sed -i 's|http://deb.debian.org/debian|http://ftp.us.debian.org/debian|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ================== Copy Python Dependencies ==================
COPY requirements.txt .

# ================== Install Python Packages ==================
RUN pip install --no-cache-dir -r requirements.txt

# ================== Copy Application Code ==================
COPY . .

# ================== Default Command ==================
CMD ["python", "photo_processor.py"]
