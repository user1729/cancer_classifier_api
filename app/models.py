from transformers import pipeline
import os


class CancerClassifier:
    def __init__(self, model_path: str):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            return_all_scores=True,
            device=0 if os.environ.get("USE_GPU", "false").lower() == "true" else -1,
        )

    def predict(self, text: str):
        results = self.classifier(text)
        return {
            "predicted_labels": ["Non-Cancer", "Cancer"],
            "confidence_scores": {
                "Non-Cancer": results[0][0]["score"],
                "Cancer": results[0][1]["score"],
            },
        }
