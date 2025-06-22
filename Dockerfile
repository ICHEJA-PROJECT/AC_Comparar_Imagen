FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean

# Create and use working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port and run app
EXPOSE 5001
CMD ["python3", "app.py"]
