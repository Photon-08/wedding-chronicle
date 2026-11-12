
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE_DIR = Path(
    "/Users/indranilbhattacharyya/Documents/photo-album-project/photo app_augmented"
)

OUTPUT_DIR = Path(
    "/Users/indranilbhattacharyya/Documents/photo-album-project/photo app_split"
)

CLASSES = ["Neel", "Indu", "Both", "Others"]

TRAIN_RATIO = 0.95
RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


# --------------------------------------------------
# 1. Identify original image groups
# --------------------------------------------------

def get_original_group(filename):
    """
    Group an original image with all its augmented variants.

    Example:
        photo.jpg
        photo_aug_01.jpg
        photo_aug_02.jpg

    All belong to the same group: photo
    """

    stem = Path(filename).stem

    # Remove augmentation suffix
    stem = re.sub(r"_aug_\d+$", "", stem)

    return stem


# --------------------------------------------------
# 2. Collect images class-wise
# --------------------------------------------------

def collect_class_images(class_name):

    folder = SOURCE_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing class folder: {folder}"
        )

    image_groups = defaultdict(list)

    for image_path in folder.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        group_name = get_original_group(image_path.name)

        image_groups[group_name].append(image_path)

    return image_groups


# --------------------------------------------------
# 3. Split groups into train and test
# --------------------------------------------------

def split_groups(image_groups):

    groups = list(image_groups.keys())

    random.shuffle(groups)

    if len(groups) < 2:
        raise ValueError(
            "At least two independent original images "
            "are required per class for a train/test split."
        )

    train_count = int(len(groups) * TRAIN_RATIO)

    # Ensure both splits contain at least one group
    train_count = max(1, min(train_count, len(groups) - 1))

    train_groups = groups[:train_count]
    test_groups = groups[train_count:]

    return train_groups, test_groups


# --------------------------------------------------
# 4. Copy images to destination
# --------------------------------------------------

def copy_groups(groups, image_groups, destination):

    copied = 0

    for group in groups:

        for image_path in image_groups[group]:

            destination_path = destination / image_path.name

            shutil.copy2(
                image_path,
                destination_path
            )

            copied += 1

    return copied


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    train_root = OUTPUT_DIR / "Train"
    test_root = OUTPUT_DIR / "Test"

    # Avoid accidentally mixing with an older split
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists and is not empty. "
            "Rename or remove it before running this script."
        )

    for class_name in CLASSES:
        (train_root / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

        (test_root / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    summary = {}

    for class_name in CLASSES:

        print(f"\nProcessing class: {class_name}")

        image_groups = collect_class_images(class_name)

        total_originals = len(image_groups)

        total_files = sum(
            len(files) for files in image_groups.values()
        )

        print("Original image groups:", total_originals)
        print("Total image files:", total_files)

        train_groups, test_groups = split_groups(
            image_groups
        )

        train_count = copy_groups(
            train_groups,
            image_groups,
            train_root / class_name
        )

        test_count = copy_groups(
            test_groups,
            image_groups,
            test_root / class_name
        )

        summary[class_name] = {
            "original_groups": total_originals,
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "train_images": train_count,
            "test_images": test_count
        }

    # --------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    total_train = 0
    total_test = 0

    for class_name, stats in summary.items():

        print(f"\n{class_name}")

        print(
            f"  Original groups : {stats['original_groups']}"
        )

        print(
            f"  Train groups    : {stats['train_groups']}"
        )

        print(
            f"  Test groups     : {stats['test_groups']}"
        )

        print(
            f"  Train images    : {stats['train_images']}"
        )

        print(
            f"  Test images     : {stats['test_images']}"
        )

        total_train += stats["train_images"]
        total_test += stats["test_images"]

    print("\n" + "=" * 60)
    print("TOTAL TRAIN IMAGES:", total_train)
    print("TOTAL TEST IMAGES :", total_test)
    print(
        "Actual train ratio:",
        round(total_train / (total_train + total_test) * 100, 2),
        "%"
    )
    print(
        "Actual test ratio:",
        round(total_test / (total_train + total_test) * 100, 2),
        "%"
    )

    print("\nDataset created at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
