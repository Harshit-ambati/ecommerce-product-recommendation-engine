"""Prepare a deployable model when generated files are absent."""
from pathlib import Path
from .model import Recommender
from .pipeline import run


def ensure_model(root):
    root = Path(root)
    artifact = root / 'artifacts' / 'recommender.pkl'
    if not artifact.is_file():
        run(root)
    return Recommender.load(artifact)
