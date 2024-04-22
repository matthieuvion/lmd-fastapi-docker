from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

"""
Convenience module to download model files (model, tokenizer, cfg files) from HuggingFace hub

- Create app/model/ dir if not existing
- Download from HF and save model, cfg saved in app/model/ dir
- Module will be called in DockerFile, during image creation
"""

BASE_DIR = Path(__file__).resolve(strict=True).parent

MODEL_DIR = BASE_DIR / "model"
MODEL_NAME = "gentilrenard/multi-e5-base_lmd-comments_q8_onnx"


def download_model(model_dir, model_name):
    """
    Download ONNX model from HF, to the specified (MODEL_DIR) directory.

    Args:
    model_dir (Path): The directory path to save the model and tokenizer.
    model_name (str): The Hugging Face model identifier.
    """
    # Check if directory exists, create "model/" dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ORTModelForSequenceClassification.from_pretrained(model_name)

    # Save the model and tokenizer to the directory
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    download_model(MODEL_DIR, MODEL_NAME)
