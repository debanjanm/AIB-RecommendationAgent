"""
Download and convert the Fashion Product Images (Small) dataset from HuggingFace.
Source: ashraq/fashion-product-images-small (MIT License)
Original: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Produces:
  - data/sample_products/images/{id}.jpg   (real product photos)
  - data/sample_products/products.json     (replaces the placeholder catalog)

Run:
  python utils/load_fashion_dataset.py
  python utils/seed_data.py                # re-seed ChromaDB with real data
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

HF_DATASET_NAME = "ashraq/fashion-product-images-small"
PRODUCTS_PER_CATEGORY = 100          # 100 × 5 categories = 500 products total
RANDOM_SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data", "sample_products", "images")
PRODUCTS_JSON = os.path.join(ROOT, "data", "sample_products", "products.json")

# ── Category mapping ──────────────────────────────────────────────────────────
# Map dataset masterCategory → our project category labels

MASTER_TO_CATEGORY = {
    "Apparel":       "apparel",
    "Footwear":      "footwear",
    "Accessories":   "accessories",
    "Personal Care": "personal_care",
    "Sporting Goods":"sporting_goods",
    "Home":          "home",
    "Free Items":    "accessories",   # fallback
}

# Which masterCategories to include (skip sparse/noisy ones)
INCLUDE_CATEGORIES = {"Apparel", "Footwear", "Accessories", "Personal Care", "Sporting Goods"}

# ── Price heuristics by masterCategory ───────────────────────────────────────

PRICE_RANGES = {
    "Apparel":       (19.99,  149.99),
    "Footwear":      (39.99,  249.99),
    "Accessories":   (14.99,  199.99),
    "Personal Care": (8.99,    59.99),
    "Sporting Goods":(24.99,  299.99),
    "Free Items":    (0.00,     0.00),
}

# ── Description templates ─────────────────────────────────────────────────────
# Built from structured metadata so every product has a meaningful description.

DESCRIPTION_TEMPLATES = [
    "{name} — a {colour} {article} designed for {gender_lower}. Perfect for {usage_lower} occasions, especially during {season_lower}. Part of the {sub} collection.",
    "Introducing the {name}, a stylish {colour} {article} tailored for {gender_lower}. Ideal for {usage_lower} wear in {season_lower}. A must-have in the {sub} category.",
    "The {name} is a versatile {colour} {article} suitable for {gender_lower}. Whether it's for {usage_lower} use or everyday styling, this {season_lower} piece from {sub} delivers both comfort and style.",
    "Elevate your wardrobe with the {name} — a {colour} {article} crafted for {gender_lower}. Designed for {usage_lower} occasions and {season_lower} seasons. Explore the {sub} range.",
]

random.seed(RANDOM_SEED)


def make_description(row: dict) -> str:
    gender = row.get("gender", "Unisex") or "Unisex"
    article = (row.get("articleType") or "item").lower()
    colour = (row.get("baseColour") or "versatile").lower()
    season = (row.get("season") or "all-season").lower()
    usage = (row.get("usage") or "everyday").lower()
    sub = (row.get("subCategory") or "General").title()
    name = row.get("productDisplayName") or "Fashion Item"
    gender_lower = gender.lower() if gender.lower() not in ("unisex", "boys", "girls") else gender.lower()

    template = random.choice(DESCRIPTION_TEMPLATES)
    return template.format(
        name=name,
        colour=colour,
        article=article,
        gender_lower=gender_lower,
        usage_lower=usage,
        season_lower=season,
        sub=sub,
    )


def make_tags(row: dict) -> list[str]:
    tags = []
    for field in ("subCategory", "articleType", "baseColour", "gender", "season", "usage"):
        val = row.get(field)
        if val and str(val).strip().lower() not in ("nan", "none", ""):
            tags.append(str(val).strip().lower().replace(" ", "-"))
    return list(dict.fromkeys(tags))  # deduplicate, preserve order


def assign_price(master_cat: str) -> float:
    low, high = PRICE_RANGES.get(master_cat, (9.99, 99.99))
    if low == high:
        return round(low, 2)
    # Non-uniform: skew toward lower prices (more realistic)
    raw = random.triangular(low, high, low + (high - low) * 0.35)
    # Round to .99 pricing
    return round(raw - 0.01, 2) + 0.99 if raw > 1 else round(raw, 2)


def save_image(pil_image, product_id: str) -> str:
    """Save PIL image to images dir, return filename."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    filename = f"{product_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        img = pil_image.convert("RGB")
        # Upscale small images to at least 256px (CLIP needs 224px)
        if max(img.size) < 256:
            scale = 256 / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        img.save(path, "JPEG", quality=88)
    return filename


def main():
    print("=" * 60)
    print("Fashion Product Images — Dataset Loader")
    print(f"Source : {HF_DATASET_NAME}")
    print(f"Target : {PRODUCTS_JSON}")
    print(f"Images : {IMAGES_DIR}")
    print("=" * 60)

    # ── Step 1: Load from HuggingFace ─────────────────────────────────────────
    print("\n[1/4] Downloading dataset from HuggingFace (first run ~271 MB)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset(HF_DATASET_NAME, split="train")
    print(f"      Loaded {len(ds):,} total records.")

    # ── Step 2: Filter + sample ───────────────────────────────────────────────
    print(f"\n[2/4] Filtering to {PRODUCTS_PER_CATEGORY} items per category...")

    # Group indices by masterCategory
    from collections import defaultdict
    cat_indices = defaultdict(list)
    for i, row in enumerate(ds):
        mc = row.get("masterCategory", "")
        if mc in INCLUDE_CATEGORIES:
            cat_indices[mc].append(i)

    print("      Available per category:")
    for cat, idxs in sorted(cat_indices.items()):
        print(f"        {cat:20s} → {len(idxs):,} items")

    selected_indices = []
    for cat, idxs in cat_indices.items():
        k = min(PRODUCTS_PER_CATEGORY, len(idxs))
        selected_indices.extend(random.sample(idxs, k))

    random.shuffle(selected_indices)
    print(f"      Selected {len(selected_indices)} products total.")

    # ── Step 3: Convert + save images ─────────────────────────────────────────
    print(f"\n[3/4] Converting records and saving images...")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    products = []
    skipped = 0

    for idx in tqdm(selected_indices, desc="Processing", unit="product"):
        row = ds[idx]

        pid_raw = row.get("id", "")
        pid = f"fashion_{pid_raw}"
        name = (row.get("productDisplayName") or "").strip()
        master = row.get("masterCategory", "Accessories")

        if not name or not pid_raw:
            skipped += 1
            continue

        # Save image
        pil_img = row.get("image")   # HuggingFace returns PIL.Image directly
        if pil_img is None:
            skipped += 1
            continue

        try:
            filename = save_image(pil_img, pid)
        except Exception as e:
            print(f"\n  [warn] Could not save image for {pid}: {e}")
            skipped += 1
            continue

        product = {
            "id": pid,
            "name": name,
            "description": make_description(row),
            "category": MASTER_TO_CATEGORY.get(master, "accessories"),
            "price": assign_price(master),
            "brand": "Fashion",
            "tags": make_tags(row),
            "image_filename": filename,
        }
        products.append(product)

    print(f"\n      Converted: {len(products)} products  |  Skipped: {skipped}")

    # ── Step 4: Write products.json ───────────────────────────────────────────
    print(f"\n[4/4] Writing {PRODUCTS_JSON}...")
    os.makedirs(os.path.dirname(PRODUCTS_JSON), exist_ok=True)
    with open(PRODUCTS_JSON, "w") as f:
        json.dump(products, f, indent=2)

    print(f"\n✅ Done! {len(products)} products saved.")
    print(f"   Images  → {IMAGES_DIR}")
    print(f"   Catalog → {PRODUCTS_JSON}")
    print("\nNext step: python utils/seed_data.py")
    print("           (clears old ChromaDB index and seeds with real fashion data)")


if __name__ == "__main__":
    main()
