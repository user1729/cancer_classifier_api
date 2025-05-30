from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import CancerClassifier
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Cancer Abstract Classifier API",
    description="API for classifying medical abstracts as cancer-related or not",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
classifier = CancerClassifier("models/fine_tuned")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify a medical abstract"""
    try:
        result = classifier.predict(request.text)
        return {
            "predicted_labels": result["predicted_labels"],
            "confidence_scores": result["confidence_scores"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
