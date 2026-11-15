import argparse
import hashlib
import json
import os

from datetime import datetime, timezone
from pathlib import Path

from src.validator import validate_all, write_report


ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "publishers.json"
OUTPUT_PATH = ROOT / "output" / "catalog.json"
CHECKSUM_PATH = ROOT / "output" / "catalog.sha256"

# 2.x: head_dim split into head_dim_k / head_dim_v, weight and expert byte
# totals added, unsupported architectures excluded, KV precision never stored.
SCHEMA_VERSION = "2.0.0"


def default_min_valid():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("min_valid_models", 1)


def generation_time():
    """UTC now, or SOURCE_DATE_EPOCH for reproducible builds."""

    epoch = os.environ.get("SOURCE_DATE_EPOCH")

    if epoch:
        return datetime.fromtimestamp(int(epoch), timezone.utc)

    return datetime.now(timezone.utc)


def content_hash(models):
    """Hash of the model records only (independent of timestamps)."""

    canonical = json.dumps(
        models, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_catalog(min_valid=None):
    if min_valid is None:
        min_valid = default_min_valid()

    models, unsupported, rejected = validate_all()

    write_report(models, unsupported, rejected)

    if len(models) < min_valid:
        raise RuntimeError(
            f"Not enough valid models: {len(models)} < {min_valid}"
        )

    models.sort(key=lambda item: (item["repo_id"], item["file_name"]))

    now = generation_time()

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": now.strftime("%Y.%m.%d.%H%M%S"),
        "generated_at": now.isoformat(),
        "model_count": len(models),
        "content_sha256": content_hash(models),
        "models": models,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = (
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    checksum = hashlib.sha256(content).hexdigest()

    # Write temporary files first, then publish both.
    temporary_catalog = OUTPUT_PATH.with_name("catalog.json.tmp")
    temporary_checksum = CHECKSUM_PATH.with_name("catalog.sha256.tmp")

    temporary_catalog.write_bytes(content)
    temporary_checksum.write_text(
        f"{checksum}  catalog.json\n", encoding="utf-8"
    )

    temporary_catalog.replace(OUTPUT_PATH)
    temporary_checksum.replace(CHECKSUM_PATH)

    print("\nCatalog generated successfully.")
    print(f"Version: {catalog['catalog_version']}")
    print(f"Models: {catalog['model_count']}")
    print(f"Unsupported (excluded): {len(unsupported)}")
    print(f"Rejected (excluded): {len(rejected)}")
    print(f"SHA-256: {checksum}")
    print(f"Catalog: {OUTPUT_PATH}")
    print(f"Checksum: {CHECKSUM_PATH}")

    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-valid",
        type=int,
        default=None,
        help="Minimum valid models (default: config min_valid_models)",
    )

    args = parser.parse_args()

    build_catalog(args.min_valid)
