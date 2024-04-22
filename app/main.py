from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

description = """
Le Monde - Ukraine invasion - comments (FR) classification API 🤔

### Model

Model:
- NN clf w/ e5-base embeddings trained on augmented/synthetic dataset (voting ensemble : Setfit fine-tuned Mistral-7B-base)
- Quantized & optimized, ONNX + inference engine/pipeline Huggingface Optimum onnxruntime
- Final model available on HF Hub : gentilrenard/multi-e5-base_lmd-comments_q8_onnx

Labels:
- **pro_ukraine**: (0) rather supportive of Ukraine.
- **pro_russia**: (1) rather supportive of Russia.
- **no_opinion**: (2) off topic or comments with no clear opinion.

### Predict

Given a text string (comment), predict user opinion on Ukraine invasion : pro_ukraine (0) pro_russia (1), no_opinion (2)
"""

app = FastAPI(
    title="LMD Comment Classification API",
    description=description,
    version="0.1.0",
    contact={
        "name": "Matthieu Vion",
        "url": "https://github.com/matthieuvion/lmd-fastapi-docker",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "http://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

MODEL_DIR = Path(__file__).resolve(strict=True).parent / "model"
MODEL_NAME = "multi-e5-base_lmd-comments_q8_onnx"


class TextIn(BaseModel):
    comment: str


class Prediction(BaseModel):
    label: str
    score: float


def load_model(model_path):
    """Load the ONNX model and tokenizer"""
    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


model, tokenizer = load_model(MODEL_DIR)


@app.get("/", summary="Status, model version", tags=["General"])
def home():
    """Landing page with status and model version"""
    return {"health_check": "ok", "model_version": app.version}


@app.post(
    "/predict",
    response_model=Prediction,
    summary="Classify comment",
    tags=["Prediction"],
)
def predict(payload: TextIn):
    """Predict endpoint for classifying comments."""
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = clf(payload.comment)
    label_map = {0: "pro_ukraine", 1: "pro_russia", 2: "no_opinion"}
    if results:
        result = results[0]
        label = label_map[int(result["label"].split("_")[-1])]
        score = result["score"]
        return {"label": label, "score": score}
    else:
        return {"label": "Error in label detection", "score": 0.0}
