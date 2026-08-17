from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/Users/vamsikrishnagondu/Desktop/Project 1")

SUIM_DIR = PROJECT_ROOT / "SUIM"

TRAIN_IMAGES_DIR = SUIM_DIR / "train_val" / "images"
TRAIN_MASKS_DIR = SUIM_DIR / "train_val" / "masks"

TEST_IMAGES_DIR = SUIM_DIR / "TEST" / "images"
TEST_MASKS_DIR = SUIM_DIR / "TEST" / "masks"


# ============================================================
# SUPPORTED FILE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_files(directory, extensions):
    """Return all files with supported extensions."""
    
    if not directory.exists():
        return []

    return sorted(
        file for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    )


# ============================================================
# DIRECTORY CHECK
# ============================================================

def check_directory(directory):
    """Check whether a required dataset directory exists."""
    
    if directory.exists():
        print(f"✓ Found: {directory}")
    else:
        print(f"✗ Missing: {directory}")


# ============================================================
# IMAGE-MASK PAIR CHECK
# ============================================================

def check_pairs(image_files, mask_files):
    """Check whether images have corresponding masks."""
    
    image_stems = {file.stem for file in image_files}
    mask_stems = {file.stem for file in mask_files}

    missing_masks = image_stems - mask_stems
    missing_images = mask_stems - image_stems

    print("\n--- Image-Mask Pair Check ---")

    print(f"Images without masks : {len(missing_masks)}")
    print(f"Masks without images : {len(missing_images)}")

    if missing_masks:
        print("\nImages missing masks:")
        for name in sorted(missing_masks)[:10]:
            print(f"  {name}")

    if missing_images:
        print("\nMasks missing images:")
        for name in sorted(missing_images)[:10]:
            print(f"  {name}")

    if not missing_masks and not missing_images:
        print("✓ All image-mask pairs are present.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SUIM DATASET EXPLORATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check required directories
    # --------------------------------------------------------

    print("\n--- Directory Check ---")

    check_directory(SUIM_DIR)
    check_directory(TRAIN_IMAGES_DIR)
    check_directory(TRAIN_MASKS_DIR)
    check_directory(TEST_IMAGES_DIR)
    check_directory(TEST_MASKS_DIR)

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    train_images = get_files(TRAIN_IMAGES_DIR, IMAGE_EXTENSIONS)
    train_masks = get_files(TRAIN_MASKS_DIR, MASK_EXTENSIONS)

    test_images = get_files(TEST_IMAGES_DIR, IMAGE_EXTENSIONS)
    test_masks = get_files(TEST_MASKS_DIR, MASK_EXTENSIONS)

    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    print("\n--- Dataset Statistics ---")

    print(f"Training/Validation Images : {len(train_images)}")
    print(f"Training/Validation Masks  : {len(train_masks)}")

    print(f"Test Images                : {len(test_images)}")
    print(f"Test Masks                 : {len(test_masks)}")

    # --------------------------------------------------------
    # Pair verification
    # --------------------------------------------------------

    check_pairs(train_images, train_masks)
    check_pairs(test_images, test_masks)

    # --------------------------------------------------------
    # Sample files
    # --------------------------------------------------------

    print("\n--- Sample Training Files ---")

    for image in train_images[:5]:
        print(f"Image: {image.name}")

    print("\n--- Sample Training Masks ---")

    for mask in train_masks[:5]:
        print(f"Mask : {mask.name}")

    # --------------------------------------------------------
    # Sample test files
    # --------------------------------------------------------

    print("\n--- Sample Test Files ---")

    for image in test_images[:5]:
        print(f"Image: {image.name}")

    print("\n--- Sample Test Masks ---")

    for mask in test_masks[:5]:
        print(f"Mask : {mask.name}")

    print("\n" + "=" * 60)
    print("DATASET EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()