# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Hugging Face expects
EXPOSE 7860

# Set environment variable for Flask
ENV PORT=7860

# Run Flask app
CMD ["python", "app.py"]
