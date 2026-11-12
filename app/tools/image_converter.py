
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm

# Enable HEIC/HEIF support in Pillow
register_heif_opener()

# Root dataset directory
ROOT_DIR = Path(
    "/Users/indranilbhattacharyya/Documents/photo-album-project/photo app"
)

# JPEG quality: 95 gives high quality with reasonable file sizes
JPEG_QUALITY = 95

# Find all HEIC/HEIF files recursively
heic_files = [
    path
    for path in ROOT_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {".heic", ".heif"}
]

print(f"Found {len(heic_files)} HEIC/HEIF files.")

converted = 0
skipped = 0
failed = 0

for heic_path in tqdm(heic_files, desc="Converting images"):

    # Save JPEG beside the original
    jpeg_path = heic_path.with_suffix(".jpg")

    # Avoid overwriting an existing JPEG
    if jpeg_path.exists():
        print(f"\nSkipping existing file: {jpeg_path.name}")
        skipped += 1
        continue

    try:
        with Image.open(heic_path) as image:

            # Apply EXIF orientation correctly
            image = ImageOps.exif_transpose(image)

            # JPEG does not support transparency
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            image.save(
                jpeg_path,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True
            )

        converted += 1

    except Exception as error:
        print(f"\nFailed: {heic_path}")
        print(f"Reason: {error}")
        failed += 1

print("\n========== Conversion Summary ==========")
print(f"Total HEIC/HEIF files : {len(heic_files)}")
print(f"Successfully converted: {converted}")
print(f"Skipped               : {skipped}")
print(f"Failed                : {failed}")
print("Original HEIC files have been preserved.")
