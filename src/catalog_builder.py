
import json
import hashlib
import argparse

from datetime import datetime, timezone
from pathlib import Path

from validator import validate_models


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = ROOT / "output" / "catalog.json"
CHECKSUM_PATH = ROOT / "output" / "catalog.sha256"


def build_catalog(min_valid=1):

    models = validate_models()

    if len(models) < min_valid:
        raise RuntimeError(
            f"Not enough valid models: {len(models)}"
        )

    # Stable ordering makes builds reproducible
    # when the underlying model records have not changed.
    models.sort(
        key=lambda item: (
            item["repo_id"],
            item["file_name"]
        )
    )

    now = datetime.now(timezone.utc)

    catalog = {
        "schema_version": "1.0.0",
        "catalog_version": now.strftime("%Y.%m.%d.%H%M%S"),
        "generated_at": now.isoformat(),
        "model_count": len(models),
        "models": models
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    content = (
        json.dumps(
            catalog,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    ).encode("utf-8")

    checksum = hashlib.sha256(content).hexdigest()

    # Write temporary files first.
    temporary_catalog = OUTPUT_PATH.with_name("catalog.json.tmp")
    temporary_checksum = CHECKSUM_PATH.with_name("catalog.sha256.tmp")

    temporary_catalog.write_bytes(content)

    temporary_checksum.write_text(
        f"{checksum}  catalog.json\n",
        encoding="utf-8"
    )

    # Publish both generated files.
    temporary_catalog.replace(OUTPUT_PATH)
    temporary_checksum.replace(CHECKSUM_PATH)

    print("\nCatalog generated successfully.")
    print(f"Version: {catalog['catalog_version']}")
    print(f"Models: {catalog['model_count']}")
    print(f"SHA-256: {checksum}")
    print(f"Catalog: {OUTPUT_PATH}")
    print(f"Checksum: {CHECKSUM_PATH}")

    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-valid",
        type=int,
        default=1
    )

    args = parser.parse_args()

    build_catalog(args.min_valid)
