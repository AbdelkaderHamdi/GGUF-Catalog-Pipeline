
import json
import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "output" / "parsed_models.json"


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    repo_id: str
    file_name: str
    download_url: HttpUrl

    file_size_bytes: int = Field(gt=0)
    quantization: str = "UNKNOWN"

    architecture: Optional[str] = None
    model_name: Optional[str] = None
    parameters: Optional[int] = None

    layers: Optional[int] = None
    attention_heads: Optional[int] = None
    kv_heads: Optional[int] = None
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None

    file_type: Optional[int] = None
    downloads: int = Field(default=0, ge=0)
    license: Optional[str] = None
    last_modified: Optional[str] = None


def stable_id(repo_id, filename):
    raw = f"{repo_id}\n{filename}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def validate_models():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        raw_models = json.load(f)

    valid_models = []
    rejected = []
    seen = set()

    for model in raw_models:

        if model.get("parse_status") != "success":
            rejected.append({
                "file": model.get("file_name"),
                "reason": "GGUF parsing failed"
            })
            continue

        try:
            normalized = {
                "id": stable_id(
                    model["repo_id"],
                    model["file_name"]
                ),
                "repo_id": model["repo_id"],
                "file_name": model["file_name"],
                "download_url": model["download_url"],
                "file_size_bytes": model["file_size_bytes"],
                "quantization": model.get(
                    "filename_quantization", "UNKNOWN"
                ),
                "architecture": model.get("architecture"),
                "model_name": model.get("model_name"),
                "parameters": model.get("parameters"),
                "layers": model.get("layers"),
                "attention_heads": model.get("attention_heads"),
                "kv_heads": model.get("kv_heads"),
                "context_length": model.get("context_length"),
                "embedding_length": model.get("embedding_length"),
                "file_type": model.get("file_type"),
                "downloads": model.get("downloads", 0),
                "license": model.get("license"),
                "last_modified": model.get("last_modified")
            }

            entry = ModelEntry.model_validate(normalized)

            key = (entry.repo_id, entry.file_name)

            if key in seen:
                rejected.append({
                    "file": entry.file_name,
                    "reason": "Duplicate repository/file"
                })
                continue

            seen.add(key)

            valid_models.append(
                entry.model_dump(mode="json")
            )

        except Exception as error:
            rejected.append({
                "file": model.get("file_name"),
                "reason": str(error)
            })

    print(f"Valid models: {len(valid_models)}")
    print(f"Rejected models: {len(rejected)}")

    for item in rejected[:20]:
        print(f"[REJECTED] {item}")

    return valid_models


if __name__ == "__main__":
    validate_models()
