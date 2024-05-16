FROM python:3.10-slim-bullseye
# Set environment variable
ENV APP_HOME=/app
# Set working dir in container to /app
WORKDIR $APP_HOME
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Install pipenv and uvicorn
RUN pip install pipenv uvicorn
# Copy pipfile and .env files into container
COPY Pipfile* ./
COPY .env ./
# Install dependencies using pipenv/pip 
RUN pipenv install && \
    pipenv run pip freeze > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt
# Read  PORT from env and make available
ARG PORT=8080
ENV PORT=$PORT
EXPOSE $PORT
# Copy current dir contents into container at /app
COPY . .
# run uvicorn server on launch
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]

