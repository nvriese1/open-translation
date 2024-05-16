#!/bin/bash

if [[ $1 == "dev" ]]; then

    echo "Running in development mode..."
    docker-compose -f docker-compose.dev.yaml down
    docker-compose -f docker-compose.dev.yaml up -d

elif [[ $1 == "prod" ]]; then

    echo "Running in production mode..."
    set -e
    # Source environment variables
    source ./backend/translation/.env

    echo "Project: $GOOGLE_CLOUD_PROJECT"
    echo "Bucket: $GOOGLE_CLOUD_BUCKET"
    echo "Location: $LOCATION"
    echo "Service: $SERVICE"
    echo "Memory: $MEMORY"
    echo "Min-Instances: $MIN_INSTANCES"
    echo "Port: $PORT"

    gcloud config set project $GOOGLE_CLOUD_PROJECT
    NOW="$(date +%Y%m%d%H%M%S)"
    IMAGE=us-central1-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$SERVICE/api:$NOW
    
    echo "Image: $IMAGE"
    docker build -f ./backend/translation/Dockerfile ./backend/translation -t $IMAGE --platform linux/amd64
    docker push $IMAGE
    gcloud run deploy --port=$PORT $SERVICE --image=$IMAGE --min-instances $MIN_INSTANCES --memory $MEMORY  --revision-suffix $NOW --timeout 20m

else
    echo "Invalid or no flag provided. Please use 'dev' or 'prod'."
    exit 1
fi