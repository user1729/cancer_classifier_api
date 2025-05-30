from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_labels: list[str]
    confidence_scores: dict[str, float]
