import spacy
from spacy.cli.download import download
import os

def ensure_model(model_name: str) -> str:
    """
    Ensure that the specified model is downloaded and ready for use.
    
    Args:
        model_name (str): The name of the model to ensure.
    
    Returns:
        str: The name of the ensured model.
    """
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Downloading...")
        download(model_name)
    return model_name 
