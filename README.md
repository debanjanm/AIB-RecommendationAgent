# AIB-RecommendationAgent

> Multimodal product recommendation powered by CLIP embeddings, ChromaDB vector search, and GPT-4o intelligent reranking — with a Gradio UI.

---

## What It Does

You type a query like **"blue casual shoes for men"** (or upload a product photo), and the agent finds the most relevant products from a catalog — not just by keyword matching, but by **understanding what you actually mean**.

It uses a 5-step AI pipeline:
1. **GPT-4o** expands your query into a richer search phrase
2. **CLIP** (a multimodal AI model) converts text + image into a shared vector
3. **ChromaDB** finds the 20 nearest products by vector similarity
4. **GPT-4o** reranks those 20 and explains why each product matches
5. **Top 10** results are shown in the UI with match scores and AI explanations

---

## Demo

```
Search: "floral summer dress for women"

→ Result 1 (94% match): Libas Women Floral Print Kurta
  "A pink floral kurta designed for women. Perfect for ethnic occasions in summer."

→ Result 2 (87% match): W Women's Wrap Dress
  "Matches the floral summer aesthetic with lightweight fabric for warm weather."
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env → set OPENROUTER_API_KEY=sk-or-...
```

### 3. Load the fashion dataset (one-time, ~3 min)
```bash
python utils/load_fashion_dataset.py
```

### 4. Seed the vector database
```bash
python utils/seed_data.py --reset
```

### 5. Launch
```bash
python app.py
# Opens at http://localhost:7860
```

---

## Dataset

Uses the **[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)** dataset via HuggingFace (`ashraq/fashion-product-images-small`) — MIT License.

| Category | Products |
|---|---|
| Apparel | 100 |
| Footwear | 100 |
| Accessories | 100 |
| Personal Care | 100 |
| Sporting Goods | 100 |
| **Total** | **500** |

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Gradio |
| Embeddings | CLIP ViT-B-32 (`open_clip`) |
| Vector Store | ChromaDB (local) |
| LLM | GPT-4o via OpenRouter |
| Language | Python 3.12 |

---

## Project Structure

```
├── app.py                        # Gradio UI
├── requirements.txt
├── services/
│   ├── embedding_service.py      # CLIP model wrapper
│   ├── vector_store.py           # ChromaDB wrapper
│   ├── llm_service.py            # GPT-4o query expansion + reranking
│   └── search_pipeline.py        # Orchestrates the full search flow
├── models/
│   ├── product.py                # Product data model
│   └── search.py                 # Search request/response models
├── utils/
│   ├── load_fashion_dataset.py   # Download + convert HuggingFace dataset
│   ├── seed_data.py              # Index products into ChromaDB
│   └── image_utils.py            # Image helpers
└── data/
    └── sample_products/
        ├── products.json         # Product catalog
        └── images/               # Product images
```

---

## Example Queries to Try

- `"slim fit white shirt for men for office"`
- `"red lipstick for evening parties"`
- `"lightweight backpack for travel"`
- `"sporty shoes for running outdoors"`
- Upload any fashion photo → find visually similar products

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
