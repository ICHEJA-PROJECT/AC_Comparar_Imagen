# Usar imagen base de Python 3.9 slim
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y procesamiento de imágenes
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer1.0-0 \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Crear usuario no privilegiado para ejecutar la aplicación
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Exponer puerto 5001
EXPOSE 5001

# Variables de entorno
ENV PYTHONPATH=/app
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Comando para ejecutar la aplicación
CMD ["python", "app/app.py"]