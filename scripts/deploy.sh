#!/bin/bash

# Build Docker image
docker build -t cancer-classifier .

# Run container locally for testing
docker run -d --name cancer-classifier -p 8000:8000 \
  -e USE_GPU=false \
  cancer-classifier

echo "API running at http://localhost:8000"
echo "Test with: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"text\":\"Your medical abstract here\"}'"