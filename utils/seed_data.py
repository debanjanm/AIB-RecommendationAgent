"""
Seed ChromaDB with sample products from products.json.
Idempotent: skips products that already exist.
Pass --reset to wipe ChromaDB first (useful after loading a new dataset).

Run:
  python utils/seed_data.py           # incremental (skip existing)
  python utils/seed_data.py --reset   # wipe + re-seed everything
"""
import argparse
import json
import os
import sys

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from models.product import Product
from services.embedding_service import get_embedding_service
from services.vector_store import VectorStore
from utils.image_utils import load_image, save_placeholder_image


PRODUCTS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_products", "products.json"
)
IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_products", "images"
)
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before seeding (use after loading a new dataset)",
    )
    args = parser.parse_args()

    print("Loading CLIP model (this may take a moment on first run)...")
    embedder = get_embedding_service(
        model_name=os.environ.get("CLIP_MODEL_NAME", "ViT-B-32"),
        pretrained=os.environ.get("CLIP_PRETRAINED", "openai"),
    )

    store = VectorStore(persist_dir=CHROMA_DIR)

    # ── Optional reset ────────────────────────────────────────────────────────
    if args.reset:
        existing = store.count()
        print(f"--reset flag set: deleting {existing} existing products from ChromaDB...")
        store.client.delete_collection("products")
        # Re-create the collection after deletion
        store.collection = store.client.get_or_create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"},
        )
        print("  Collection cleared.")

    with open(PRODUCTS_JSON, "r") as f:
        raw_products = json.load(f)

    total_in_catalog = len(raw_products)
    print(f"\nFound {total_in_catalog} products in catalog.")
    print(f"Seeding into ChromaDB at: {CHROMA_DIR}\n")
    seeded = 0
    skipped = 0
    errors = 0

    for i, raw in enumerate(raw_products, 1):
        product = Product(**raw)

        if store.product_exists(product.id):
            skipped += 1
            continue

        # Load or generate image
        image_path = os.path.join(IMAGES_DIR, product.image_filename)
        image = load_image(image_path)

        if image is None:
            print(f"  [placeholder] {product.id}: generating image for '{product.name}'")
            save_placeholder_image(
                product.id, product.name, product.category, IMAGES_DIR
            )
            image = load_image(image_path)

        try:
            # Generate combined embedding
            embedding = embedder.embed_product(text=product.text_blob, image=image)
            store.add_product(product, embedding)
            seeded += 1
            if seeded % 50 == 0:
                print(f"  [{i}/{total_in_catalog}] {seeded} seeded so far...")
        except Exception as e:
            print(f"  [error] {product.id}: {e}")
            errors += 1

    total = store.count()
    print(f"\n{'='*50}")
    print(f"Seeded : {seeded}")
    print(f"Skipped: {skipped}")
    print(f"Errors : {errors}")
    print(f"Total in ChromaDB: {total}")
    print(f"{'='*50}")
    print("\nRun: python app.py  →  http://localhost:7860")


if __name__ == "__main__":
    main()
