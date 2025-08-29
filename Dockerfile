# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]