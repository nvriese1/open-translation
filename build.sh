#!/bin/bash

if [[ $1 == "dev" ]]; then

    echo "Installing dependencies..."
    pip install -r backend/translation/requirements.txt
    
    export PORT=8080
    echo "Building development images..."
    docker-compose -f docker-compose.dev.yaml build

elif [[ $1 == "prod" ]]; then

    echo "No need to build for prod, just run ./deploy.sh prod"

else
    echo "Invalid or no flag provided. Please use 'dev' or 'prod'."
    exit 1
fi