# Deep Dive Documentation — AIB-RecommendationAgent

This document explains how everything works, in plain English — no jargon assumed.

---

## The Big Picture

Imagine you walk into a huge clothing store with 500 items. You tell the sales assistant:
> *"I'm looking for something casual and blue for a beach day."*

A bad assistant searches for the word "blue" and shows you anything blue — including blue office shirts and blue curtains.

A **good** assistant understands what you *mean* — casual, summery, beachy, relaxed — and brings you the right things even if none of them are labelled "beach".

That's what this project does. It understands meaning, not just words.

---

## How a Search Actually Works — Step by Step

Let's say you search for **"comfortable running shoes for flat feet"**.

### Step 1 — GPT-4o Reads Your Mind

Your query goes to GPT-4o first. GPT-4o doesn't search anything yet — it just *thinks* about what you want.

It rewrites your query into something richer:
```
Original : "comfortable running shoes for flat feet"
Expanded : "ergonomic running shoes arch support flat feet pronation cushioning stability"
```

It also extracts structured intent:
```
Category       : footwear
Key attributes : arch support, cushioning, stability
Price range    : mid-range
```

This step is important because CLIP (the embedding model) works much better with detailed, descriptive phrases than short user queries.

**File:** `services/llm_service.py` → `expand_query()`

---

### Step 2 — CLIP Converts Text to Numbers

CLIP (Contrastive Language–Image Pretraining) is an AI model trained by OpenAI on hundreds of millions of image-text pairs. It learned that a photo of a running shoe and the phrase "running shoe" should be *close together* in a mathematical space.

Your expanded query gets converted into a list of 512 numbers — called an **embedding** or **vector**. Think of it as a coordinate in a 512-dimensional space. Similar concepts end up near each other in this space.

```
"ergonomic running shoes arch support" → [0.021, -0.14, 0.087, ... 512 numbers]
```

If you also uploaded an image, CLIP converts that image into 512 numbers too, and the two are blended:
```
Final query vector = 60% text vector + 40% image vector
```

The text gets more weight because GPT-4o's expansion already made it very descriptive.

**File:** `services/embedding_service.py`

---

### Step 3 — ChromaDB Finds the Nearest Products

Every product in the catalog was already converted to 512 numbers at seed time (text + image combined, 50/50). These are stored in ChromaDB — a local vector database.

ChromaDB now does one thing: find the 20 products whose 512-number vectors are *closest* to your query vector. "Close" here means the angle between them is small — called **cosine similarity**.

```
Your query vector          →  [0.021, -0.14, 0.087, ...]
Brooks Adrenaline vector   →  [0.019, -0.13, 0.091, ...]   ← very close → 89% similar
Office Shirt vector        →  [-0.12, 0.44, -0.21, ...]    ← far away   → 23% similar
```

This happens in milliseconds even with thousands of products.

**File:** `services/vector_store.py`

---

### Step 4 — GPT-4o Reranks the Results

The 20 nearest products from ChromaDB are good candidates, but vector similarity isn't perfect. A "flat white sneaker" might be mathematically close to "running shoe" but isn't actually what you want.

GPT-4o reads your original query, your inferred intent, and a summary of all 20 candidates — and reranksthem by *semantic* relevance. It also writes a one-sentence explanation for each:

```json
[
  { "id": "foot_001", "score": 0.94, "match_reason": "Specifically designed for flat feet with arch support and maximum cushioning." },
  { "id": "foot_005", "score": 0.87, "match_reason": "High-performance running shoe with excellent stability features." },
  ...
]
```

**File:** `services/llm_service.py` → `rerank()`

---

### Step 5 — Results Are Shown in Gradio

The top 10 reranked products come back to the UI. Each card shows:
- Product image
- Name, brand, price
- Match % badge (green = 80%+, amber = 60–80%, grey = below 60%)
- GPT-4o's match explanation
- Tags

**File:** `app.py`

---

## How Products Are Indexed (Seeding)

Before anyone can search, every product needs to be converted into a vector and stored. This is the seeding step.

For each product:
1. A text blob is constructed: `"Adidas Ultraboost. Running. Boost cushioning. Tags: running, road, energy-return"`
2. CLIP converts this to a 512-d text vector
3. The product image is loaded and CLIP converts it to a 512-d image vector
4. These two are combined: `0.5 × text + 0.5 × image`, then re-normalized
5. The combined vector + all metadata (name, price, category etc.) is stored in ChromaDB

**Files:** `utils/seed_data.py`, `services/embedding_service.py`, `services/vector_store.py`

---

## The Dataset

We use the **Fashion Product Images (Small)** dataset from HuggingFace (`ashraq/fashion-product-images-small`).

Original dataset has 44,000+ products. We sample 500 (100 per category) to keep seeding fast for a demo.

The dataset has:
- A real product image per item (small resolution, ~60px — we upscale to 256px)
- Metadata: `productDisplayName`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `gender`, `season`, `usage`

What it **doesn't** have (so we generate it):
- **Description** → built from a template: *"A {colour} {articleType} for {gender}. Ideal for {usage} in {season}."*
- **Price** → randomly assigned by category range using a realistic distribution
- **Brand** → set to `"Fashion"` (not in dataset)

**File:** `utils/load_fashion_dataset.py`

---

## What Happens When You Upload an Image

If you upload an image alongside your text query:

1. Gradio sends the PIL image object to the search pipeline
2. CLIP encodes it to a 512-d image vector
3. CLIP also encodes your (GPT-4o expanded) text to a 512-d text vector
4. They're blended: `60% text + 40% image` (configurable via the slider)
5. The combined vector is used to search ChromaDB

This means you can do things like:
- Upload a shoe photo + type "but in black" → finds visually similar shoes, biased toward black
- Upload nothing + type a query → pure text semantic search
- Upload an image + leave text empty → pure visual similarity search

---

## File Map — What Each File Does

| File | Plain English |
|---|---|
| `app.py` | The Gradio web UI. Handles user input, calls the pipeline, renders results as HTML cards |
| `services/search_pipeline.py` | The conductor — calls GPT-4o, then CLIP, then ChromaDB, then GPT-4o again, returns results |
| `services/embedding_service.py` | Loads CLIP once at startup. Converts text or images into 512-number vectors |
| `services/vector_store.py` | Talks to ChromaDB. Stores and retrieves product vectors |
| `services/llm_service.py` | Talks to GPT-4o via OpenRouter. Does query expansion and reranking |
| `models/product.py` | Defines what a Product looks like (name, price, tags, etc.) |
| `models/search.py` | Defines what a search request and response look like |
| `utils/load_fashion_dataset.py` | Downloads the HuggingFace dataset, converts it, saves images and products.json |
| `utils/seed_data.py` | Reads products.json, generates embeddings, stores everything in ChromaDB |
| `utils/image_utils.py` | Helper functions for loading, resizing, and encoding images |
| `data/sample_products/products.json` | The product catalog — 500 fashion items |
| `data/sample_products/images/` | Product images saved as JPEGs |
| `data/chroma_db/` | ChromaDB's storage folder (auto-created, git-ignored) |

---

## Where the AI Happens

Three AI models are used:

| Model | Where | What it does |
|---|---|---|
| **CLIP ViT-B-32** | Runs locally on your machine | Converts text and images into comparable vectors |
| **GPT-4o** (via OpenRouter) | API call | Expands queries, reranks results, writes explanations |
| *(ChromaDB)* | Local | Not AI itself — just stores and searches vectors efficiently |

CLIP is downloaded once (~600 MB) and cached by PyTorch. Every search runs CLIP locally — fast and free.

GPT-4o is called twice per search (query expansion + reranking) — costs a fraction of a cent per search via OpenRouter.

---

## What Could Go Wrong & How It's Handled

| Failure | What happens |
|---|---|
| OpenRouter API key missing | `KeyError` at startup — set `OPENROUTER_API_KEY` in `.env` |
| GPT-4o call fails / times out | Graceful fallback: uses original query (no expansion) and sorts by cosine similarity only |
| GPT-4o returns bad JSON | Same fallback — logs a warning, doesn't crash |
| Product image missing | Generates a coloured placeholder image with the product name on it |
| ChromaDB empty (not seeded) | Returns 0 results with a helpful message |
| Image too small for CLIP | Pillow upscales to 256px before CLIP sees it |

---

## Next Steps

Here are natural improvements, roughly in order of effort:

### 🟢 Easy (a few hours)
- **Real prices** — add a price field to the dataset by scraping or using a price lookup API
- **Real brand names** — extract from `productDisplayName` (many include the brand)
- **Filter by category in the UI** — add a Gradio dropdown to pre-filter results by category
- **Pagination** — show "Load more" to fetch results 11–20
- **Dark mode** — Gradio supports it natively with `gr.themes.Soft()`

### 🟡 Medium (a day or two)
- **GPT-4o generated descriptions** — replace template descriptions with richer ones generated by GPT-4o from the metadata (much better embedding quality)
- **Use the full 44k dataset** — takes ~2–3 hours to seed but gives much richer search results
- **Add a "Find Similar" button** on each product card — calls `search_pipeline.find_similar(product_id)`
- **Hybrid search** — combine vector search with keyword (BM25) search using ChromaDB's `where_document` filter for better recall
- **Caching** — cache GPT-4o expansion results for identical queries to save API costs

### 🔴 Advanced (multi-day)
- **FastAPI backend** — expose the search pipeline as a REST API so any frontend can consume it
- **React frontend** — replace Gradio with a custom React UI for full design control
- **User feedback loop** — thumbs up/down on results → fine-tune reranking weights
- **Pinecone or Qdrant** — swap ChromaDB for a managed vector DB for production scale
- **Multimodal reranking** — send product images (not just text summaries) to GPT-4o Vision for even better reranking
- **Personalization** — track user click history, build a user embedding, blend it into the query vector
- **A/B testing** — compare different text/image weight ratios or reranking prompts

---

## Glossary

| Term | Plain English |
|---|---|
| **Embedding / Vector** | A list of numbers that represents the meaning of a piece of text or an image. Similar things have similar numbers. |
| **CLIP** | An AI model that puts images and text into the same number space, so you can compare them directly. |
| **Vector Database** | A database optimized for storing and searching vectors by similarity (not by exact match). ChromaDB is one. |
| **Cosine Similarity** | A way to measure how similar two vectors are, regardless of their size. 1.0 = identical, 0.0 = unrelated. |
| **Query Expansion** | Rewriting a short user query into a longer, richer phrase to improve search results. |
| **Reranking** | Taking a set of initial results and re-ordering them using a smarter (but slower) model. |
| **Seeding** | Pre-processing all products into vectors and storing them in ChromaDB, so searches are fast later. |
| **HuggingFace** | A platform that hosts open-source AI models and datasets. We download our fashion dataset from there. |
| **OpenRouter** | A service that gives you access to GPT-4o (and other LLMs) via a single API key. |
