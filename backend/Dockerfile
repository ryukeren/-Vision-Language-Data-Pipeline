FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# UPGRADE PIP FIRST before installing the requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]