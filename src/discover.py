import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "publishers.json"
OUTPUT_PATH = ROOT / "output" / "discovered.json"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_license(model):
    """Read licence from model-card metadata or Hub tags.

    A card that lists several licences is ambiguous, so it is treated as
    having no usable licence rather than silently picking one.
    """

    card_data = getattr(model, "card_data", None)

    if card_data:
        value = getattr(card_data, "license", None)

        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) == 1 else None

        if value:
            return str(value).lower()

    for tag in getattr(model, "tags", None) or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()

    return None


def evaluate_repository(model, publishers, min_downloads, cutoff,
                        accepted_licenses):
    """Apply every repository filter.

    Returns (record, None) when accepted, or (None, reason) when rejected.
    Pure function so the filters can be tested without network access.
    """

    repo_id = model.id

    if not repo_id or "/" not in repo_id:
        return None, "invalid repo id"

    author = repo_id.split("/")[0].lower()

    if author not in publishers:
        return None, "publisher not allowed"

    downloads = model.downloads or 0

    if downloads < min_downloads:
        return None, "too few downloads"

    last_modified = model.last_modified

    if not last_modified or last_modified < cutoff:
        return None, "too old"

    license_name = extract_license(model)

    if license_name not in accepted_licenses:
        return None, f"licence not accepted: {license_name}"

    return {
        "repo_id": repo_id,
        "publisher": author,
        "downloads": downloads,
        "last_modified": last_modified.isoformat(),
        "license": license_name,
        "tags": model.tags or [],
    }, None


def discover_models(limit=None):
    config = load_config()

    # Sorted: iteration order of a set is not deterministic.
    publishers = sorted({p.lower() for p in config["publishers"]})

    min_downloads = config["min_downloads"]
    accepted_licenses = {x.lower() for x in config["accepted_licenses"]}
    max_repositories = limit or config["max_repositories"]
    max_scan = config.get("max_scan_per_publisher", 1000)

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=config["max_age_days"]
    )

    api = HfApi()

    print("Querying Hugging Face GGUF models...")

    results = []
    seen = set()
    reasons = {}

    for publisher in publishers:
        print(f"Discovering: {publisher}")

        # Stream repositories sorted by downloads and apply the filters
        # while streaming. The previous code truncated to N repositories
        # *before* filtering, so licence/age filters could leave far fewer
        # than N even when more eligible repositories existed.
        stream = api.list_models(
            author=publisher,
            filter="gguf",
            sort="downloads",
            full=True,
            cardData=True,
        )

        accepted_here = 0

        for model in islice(stream, max_scan):
            if model.id in seen:
                continue

            seen.add(model.id)

            record, reason = evaluate_repository(
                model, set(publishers), min_downloads, cutoff,
                accepted_licenses,
            )

            if record is None:
                key = reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue

            results.append(record)
            accepted_here += 1
            print(f"[ACCEPTED] {record['repo_id']}")

            if accepted_here >= max_repositories:
                break

    # Sort and cap once, after collecting. Ties broken by repo id so the
    # output is deterministic.
    results.sort(key=lambda r: (-r["downloads"], r["repo_id"]))
    results = results[:max_repositories]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDiscovery completed.")
    print(f"Repositories accepted: {len(results)}")
    print(f"Rejected by reason: {reasons}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum repositories to retrieve",
    )

    args = parser.parse_args()

    discover_models(args.limit)
