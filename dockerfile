# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask scikit-learn pandas joblib

# Train the model when building image
RUN python train_diabetes_model.py

# Expose port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
