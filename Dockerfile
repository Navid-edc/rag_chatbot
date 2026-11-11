# Use lightweight official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code into the container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn workers in production mode
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
