
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "publishers.json"
OUTPUT_PATH = ROOT / "output" / "discovered.json"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_license(model):
    """Read licence from model-card metadata or Hub tags."""

    card_data = getattr(model, "card_data", None)

    if card_data:
        license_value = getattr(card_data, "license", None)

        if license_value:
            return str(license_value).lower()

    tags = getattr(model, "tags", []) or []

    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()

    return None


def discover_models(limit=None):
    config = load_config()

    publishers = {
        publisher.lower()
        for publisher in config["publishers"]
    }

    min_downloads = config["min_downloads"]
    max_age_days = config["max_age_days"]

    accepted_licenses = {
        license.lower()
        for license in config["accepted_licenses"]
    }

    max_repositories = limit or config["max_repositories"]

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=max_age_days
    )

    api = HfApi()

    print("Querying Hugging Face GGUF models...")

    models = []

    for publisher in publishers:
        print(f"Discovering: {publisher}")

        publisher_models = api.list_models(
            author=publisher,
            filter="gguf",
            sort="downloads",
            limit=max_repositories,
            full=True,
            cardData=True
        )

        models.extend(publisher_models)

    results = []
    seen = set()

    for model in models:
        repo_id = model.id

        if not repo_id or repo_id in seen:
            continue

        seen.add(repo_id)

        # Publisher filter
        author = repo_id.split("/")[0].lower()

        if author not in publishers:
            continue

        # Download filter
        downloads = model.downloads or 0

        if downloads < min_downloads:
            continue

        # Recency filter
        last_modified = model.last_modified

        if not last_modified or last_modified < cutoff:
            continue

        # Licence filter
        license_name = extract_license(model)

        if license_name not in accepted_licenses:
            continue

        results.append({
            "repo_id": repo_id,
            "publisher": author,
            "downloads": downloads,
            "last_modified": last_modified.isoformat(),
            "license": license_name,
            "tags": model.tags or []
        })
        results.sort(
        key=lambda item: item["downloads"],
        reverse=True
        )

        results = results[:max_repositories]

        print(f"[ACCEPTED] {repo_id}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nDiscovery completed.")
    print(f"Repositories accepted: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum repositories to retrieve"
    )

    args = parser.parse_args()

    discover_models(args.limit)
