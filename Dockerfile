FROM python:3.10-slim

WORKDIR /app

# Keep dependency installation separate so Docker can reuse the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project after dependencies are in place.
COPY . .

# Hugging Face Spaces expects the app to listen on 7860.
EXPOSE 7860

# Serve the FastAPI app directly with uvicorn.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
