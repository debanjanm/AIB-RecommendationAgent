"""
Seed ChromaDB with sample products from products.json.
Idempotent: skips products that already exist.
Run: python utils/seed_data.py
"""
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
    print("Loading CLIP model (this may take a moment on first run)...")
    embedder = get_embedding_service(
        model_name=os.environ.get("CLIP_MODEL_NAME", "ViT-B-32"),
        pretrained=os.environ.get("CLIP_PRETRAINED", "openai"),
    )

    store = VectorStore(persist_dir=CHROMA_DIR)

    with open(PRODUCTS_JSON, "r") as f:
        raw_products = json.load(f)

    print(f"Found {len(raw_products)} products in catalog.")
    seeded = 0
    skipped = 0

    for raw in raw_products:
        product = Product(**raw)

        if store.product_exists(product.id):
            print(f"  [skip] {product.id}: {product.name}")
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

        # Generate combined embedding
        embedding = embedder.embed_product(text=product.text_blob, image=image)

        store.add_product(product, embedding)
        print(f"  [added] {product.id}: {product.name}")
        seeded += 1

    total = store.count()
    print(f"\nDone. Seeded: {seeded}, Skipped: {skipped}. Total in ChromaDB: {total}")


if __name__ == "__main__":
    main()
