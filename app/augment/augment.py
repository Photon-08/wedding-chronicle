
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path(
    "/Users/indranilbhattacharyya/Documents/photo-album-project/photo app"
)

OUTPUT_DIR = Path(
    "/Users/indranilbhattacharyya/Documents/photo-album-project/photo app_augmented"
)

# Number of NEW augmented images per original image
AUGMENTATIONS_PER_IMAGE = 5

# Only these folders will be augmented
TARGET_CLASSES = ["Neel", "Indu", "Both"]

# Image formats supported
VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

JPEG_QUALITY = 95

random.seed(42)
np.random.seed(42)


# --------------------------------------------------
# 1. Image transformations
# --------------------------------------------------

def adjust_warm_lighting(image):
    """
    Simulate warm indoor LED / tungsten lighting.
    Adds a subtle yellow-orange colour temperature.
    """

    image = np.asarray(image).astype(np.float32)

    # Warm colour cast
    image[:, :, 0] *= random.uniform(1.03, 1.15)
    image[:, :, 1] *= random.uniform(1.00, 1.06)
    image[:, :, 2] *= random.uniform(0.88, 0.98)

    image = np.clip(image, 0, 255)

    return Image.fromarray(image.astype(np.uint8))


def adjust_low_light(image):
    """
    Simulate evening indoor exposure.
    """

    brightness = random.uniform(0.55, 0.90)
    contrast = random.uniform(0.90, 1.15)

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    return image


def adjust_gamma(image):
    """
    Gamma adjustment to simulate different exposure conditions.
    """

    gamma = random.uniform(0.75, 1.35)

    lookup = [
        min(255, int(((value / 255.0) ** gamma) * 255))
        for value in range(256)
    ]

    return image.point(lookup * 3)


def add_uneven_lighting(image):
    """
    Simulate uneven household lighting and soft shadows.

    Uses a smooth illumination gradient rather than harsh
    artificial black patches.
    """

    arr = np.asarray(image).astype(np.float32)

    height, width, _ = arr.shape

    # Random direction and location of the light source
    center_x = random.uniform(0.2, 0.8) * width
    center_y = random.uniform(0.2, 0.8) * height

    yy, xx = np.mgrid[0:height, 0:width]

    distance = np.sqrt(
        ((xx - center_x) / width) ** 2 +
        ((yy - center_y) / height) ** 2
    )

    strength = random.uniform(0.08, 0.22)
    spread = random.uniform(0.4, 0.9)

    illumination = 1.0 - strength * np.exp(
        -(distance ** 2) / spread
    )

    arr *= illumination[:, :, None]

    arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def add_sensor_noise(image):
    """
    Simulate mild camera sensor noise in low-light images.
    """

    arr = np.asarray(image).astype(np.float32)

    noise_level = random.uniform(1.5, 5.0)

    noise = np.random.normal(
        loc=0,
        scale=noise_level,
        size=arr.shape
    )

    arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def add_soft_blur(image):
    """
    Simulate mild focus softness or slight motion blur.
    """

    if random.random() < 0.30:
        radius = random.uniform(0.2, 0.7)
        image = image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )

    return image


def augment_image(image):
    """
    Apply a random combination of realistic indoor
    evening photography transformations.
    """

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Random horizontal flip
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Warm indoor colour temperature
    if random.random() < 0.85:
        image = adjust_warm_lighting(image)

    # Evening exposure
    if random.random() < 0.80:
        image = adjust_low_light(image)

    # Exposure variation
    if random.random() < 0.60:
        image = adjust_gamma(image)

    # Uneven household illumination
    if random.random() < 0.60:
        image = add_uneven_lighting(image)

    # Camera sensor noise
    if random.random() < 0.65:
        image = add_sensor_noise(image)

    # Mild focus softness
    image = add_soft_blur(image)

    # Small colour and saturation variations
    if random.random() < 0.50:
        image = ImageEnhance.Color(image).enhance(
            random.uniform(0.85, 1.15)
        )

    return image


# --------------------------------------------------
# 2. Process dataset
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_originals = 0
    total_augmented = 0
    failed = 0

    for class_name in TARGET_CLASSES:

        source_folder = INPUT_DIR / class_name
        output_folder = OUTPUT_DIR / class_name

        if not source_folder.exists():
            print(f"Missing folder: {source_folder}")
            continue

        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = [
            path for path in source_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ]

        print(f"\nProcessing {class_name}: {len(image_files)} images")

        for image_path in tqdm(image_files, desc=class_name):

            try:

                with Image.open(image_path) as source:
                    original = ImageOps.exif_transpose(source).convert("RGB")

                # Save original image in output dataset
                original_name = image_path.stem + ".jpg"
                original_output = output_folder / original_name

                if not original_output.exists():
                    original.save(
                        original_output,
                        "JPEG",
                        quality=JPEG_QUALITY
                    )

                total_originals += 1

                # Generate augmented versions
                for index in range(AUGMENTATIONS_PER_IMAGE):

                    augmented = augment_image(original.copy())

                    output_name = (
                        f"{image_path.stem}_aug_{index + 1:02d}.jpg"
                    )

                    output_path = output_folder / output_name

                    augmented.save(
                        output_path,
                        "JPEG",
                        quality=JPEG_QUALITY
                    )

                    total_augmented += 1

            except Exception as error:
                print(f"\nFailed: {image_path.name}")
                print("Reason:", error)
                failed += 1

    print("\n========== DATASET SUMMARY ==========")
    print("Original images processed:", total_originals)
    print("Augmented images created:", total_augmented)
    print("Failed images:", failed)
    print("Total output images:", total_originals + total_augmented)
    print("Output directory:", OUTPUT_DIR)
    print("\nOriginal dataset remains unchanged.")


if __name__ == "__main__":
    main()
