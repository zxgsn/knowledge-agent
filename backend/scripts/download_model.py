"""Download models for local inference.

Usage:
    python scripts/download_model.py                  # download all models
    python scripts/download_model.py embedding        # download embedding model only
    python scripts/download_model.py reranker         # download reranker model only

Models are saved to backend/models/<model_name>/ so they can be loaded
without network access at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    "embedding": {
        "hf_name": "BAAI/bge-m3",
        "description": "Bi-encoder embedding model (1024-dim, multilingual, ~2.2GB)",
    },
    "reranker": {
        "hf_name": "BAAI/bge-reranker-v2-m3",
        "description": "Cross-encoder reranker (multilingual, ~568MB)",
    },
}


def download_model(key: str) -> None:
    info = MODELS[key]
    hf_name = info["hf_name"]
    local_dir = MODELS_DIR / hf_name.split("/")[-1]

    if local_dir.is_dir() and (local_dir / "config.json").exists():
        print(f"[{key}] Already exists at {local_dir}")
        return

    print(f"[{key}] Downloading {hf_name} ...")
    print(f"         {info['description']}")
    print(f"         Saving to: {local_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)

    if key == "embedding":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(hf_name)
        model.save_pretrained(str(local_dir))
    else:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(hf_name)
        model.save(str(local_dir))

    print(f"[{key}] Done. Model saved to {local_dir}")


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for key in targets:
        if key not in MODELS:
            print(f"Unknown model: {key}. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
        download_model(key)


if __name__ == "__main__":
    main()
