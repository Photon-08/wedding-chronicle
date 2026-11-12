
import argparse
import csv
import hashlib
import io
import random
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from PIL import Image
from tqdm import tqdm


API_URL = "https://commons.wikimedia.org/w/api.php"

# Queries are deliberately varied to avoid a dataset of nearly identical images.
SEARCH_TOPICS = {
    "people": [
        "people portrait photograph",
        "group of people photograph",
        "people walking street",
        "family group photograph",
        "crowd public event",
        "people working office",
        "people talking outdoors",
        "people sitting restaurant",
        "people at market",
        "people playing sports",
    ],
    "objects": [
        "everyday objects photograph",
        "food photography",
        "animals wildlife photograph",
        "cars vehicles photograph",
        "buildings architecture photograph",
        "nature landscape photograph",
        "furniture interior photograph",
        "flowers plants photograph",
        "street photography",
        "household appliances photograph",
    ],
}

HEADERS = {
    "User-Agent": "LocalFaceDatasetBuilder/1.0 (educational dataset collection)"
}

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

MIN_WIDTH = 160
MIN_HEIGHT = 160
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB


def clean_filename(name):
    name = unquote(name)
    name = re.sub(r"[^\w.-]+", "_", name)
    return name[:100]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_existing_hashes(output_dir):
    hashes = set()

    for path in output_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                hashes.add(sha256_bytes(path.read_bytes()))
            except OSError:
                pass

    return hashes


def search_commons(query, limit=50):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1200,
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if "query" not in data:
        return []

    return list(data["query"]["pages"].values())


def extract_metadata(page):
    image_info = page.get("imageinfo", [{}])[0]
    metadata = image_info.get("extmetadata", {})

    def value(key):
        item = metadata.get(key, {})
        return re.sub(r"<[^>]+>", "", item.get("value", ""))

    return {
        "title": page.get("title", ""),
        "description_url": image_info.get("descriptionurl", ""),
        "artist": value("Artist"),
        "license": value("LicenseShortName"),
        "license_url": value("LicenseUrl"),
        "credit": value("Credit"),
        "mime": image_info.get("mime", ""),
        "width": image_info.get("width", ""),
        "height": image_info.get("height", ""),
        "download_url": image_info.get("thumburl")
            or image_info.get("url", ""),
    }


def download_image(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=45,
    )
    response.raise_for_status()

    if len(response.content) > MAX_FILE_SIZE:
        return None

    try:
        image = Image.open(io.BytesIO(response.content))
        image.verify()

        # Reopen after verify to inspect dimensions.
        image = Image.open(io.BytesIO(response.content))

        if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
            return None

        if image.format not in {"JPEG", "PNG", "WEBP"}:
            return None

        # Convert unusual modes into a standard RGB image.
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        extension = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }[image.format]

        output = io.BytesIO()
        image.save(output, format=image.format)

        return output.getvalue(), extension

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download public photographs into the Others dataset folder."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to your Google Drive-synced Others folder",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of images to download (default: 200)",
    )

    parser.add_argument(
        "--people-ratio",
        type=float,
        default=0.60,
        help="Fraction of images from people-related searches (default: 0.60)",
    )

    args = parser.parse_args()

    if not 0 <= args.people_ratio <= 1:
        raise ValueError("--people-ratio must be between 0 and 1")

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"

    existing_hashes = load_existing_hashes(output_dir)
    existing_titles = set()

    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                existing_titles.add(row.get("title", ""))

    people_target = round(args.count * args.people_ratio)
    object_target = args.count - people_target

    quotas = {
        "people": people_target,
        "objects": object_target,
    }

    downloaded = {
        "people": 0,
        "objects": 0,
    }

    # Shuffle query order so each run has some variety.
    topics = {
        category: random.sample(queries, len(queries))
        for category, queries in SEARCH_TOPICS.items()
    }

    file_exists = manifest_path.exists()

    with open(
        manifest_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as manifest_file:

        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "filename",
                "category",
                "title",
                "description_url",
                "artist",
                "license",
                "license_url",
                "credit",
                "sha256",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for category, queries in topics.items():

            target = quotas[category]

            if target == 0:
                continue

            print(f"\nCollecting {target} images from: {category}")

            candidates = []

            for query in queries:
                try:
                    results = search_commons(query)

                    for page in results:
                        metadata = extract_metadata(page)

                        if not metadata["download_url"]:
                            continue

                        if metadata["mime"] not in ALLOWED_MIMES:
                            continue

                        if metadata["title"] in existing_titles:
                            continue

                        candidates.append(metadata)

                    print(
                        f"Query '{query}': "
                        f"{len(results)} results"
                    )

                    time.sleep(0.5)

                except requests.RequestException as error:
                    print(f"Search failed: {error}")

            # Randomize candidates rather than taking search ranking order.
            random.shuffle(candidates)

            for metadata in tqdm(
                candidates,
                desc=f"Downloading {category}",
            ):

                if downloaded[category] >= target:
                    break

                title = metadata["title"]

                if title in existing_titles:
                    continue

                try:
                    result = download_image(metadata["download_url"])

                    if result is None:
                        continue

                    image_bytes, extension = result
                    image_hash = sha256_bytes(image_bytes)

                    if image_hash in existing_hashes:
                        continue

                    base_name = clean_filename(
                        Path(title).stem
                    )

                    filename = (
                        f"{category}_{downloaded[category] + 1:04d}_"
                        f"{base_name}{extension}"
                    )

                    file_path = output_dir / filename

                    if file_path.exists():
                        continue

                    file_path.write_bytes(image_bytes)

                    writer.writerow({
                        "filename": filename,
                        "category": category,
                        "title": title,
                        "description_url": metadata["description_url"],
                        "artist": metadata["artist"],
                        "license": metadata["license"],
                        "license_url": metadata["license_url"],
                        "credit": metadata["credit"],
                        "sha256": image_hash,
                    })

                    manifest_file.flush()

                    existing_hashes.add(image_hash)
                    existing_titles.add(title)
                    downloaded[category] += 1

                    time.sleep(0.2)

                except requests.RequestException:
                    continue

                except OSError as error:
                    print(f"Could not save image: {error}")

            if downloaded[category] < target:
                print(
                    f"Could only collect {downloaded[category]} "
                    f"of {target} requested images for {category}."
                )

    print("\nDownload complete!")
    print("People:", downloaded["people"])
    print("Objects:", downloaded["objects"])
    print("Total:", sum(downloaded.values()))
    print("Output:", output_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
