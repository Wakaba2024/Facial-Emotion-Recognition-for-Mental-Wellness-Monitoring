# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Run Flask with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
