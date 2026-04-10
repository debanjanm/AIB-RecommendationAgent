"""
Gradio UI for the Product Recommendation Agent.
Run: python app.py
Opens at http://localhost:7860
"""
import os
import sys
import base64
import io

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image

from services.embedding_service import get_embedding_service
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.search_pipeline import SearchPipeline
from models.search import SearchResponse, RankedProduct

# ── Startup: load singletons ──────────────────────────────────────────────────

CLIP_MODEL = os.environ.get("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db")
IMAGES_DIR = os.environ.get("PRODUCT_IMAGES_DIR", "./data/sample_products/images")
TOP_K_CANDIDATES = int(os.environ.get("TOP_K_CANDIDATES", 20))
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", 10))

print("Initializing services...")
embedder = get_embedding_service(model_name=CLIP_MODEL, pretrained=CLIP_PRETRAINED)
store = VectorStore(persist_dir=CHROMA_DIR)
llm = LLMService()
pipeline = SearchPipeline(
    embedder=embedder,
    store=store,
    llm=llm,
    top_k_candidates=TOP_K_CANDIDATES,
    top_k_results=TOP_K_RESULTS,
)
print(f"Ready. ChromaDB has {store.count()} products.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_product_image_html(image_filename: str, product_name: str) -> str:
    """Return an <img> tag for a product image, either file or placeholder."""
    image_path = os.path.join(IMAGES_DIR, image_filename)
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = image_filename.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return f'<img src="data:{mime};base64,{b64}" alt="{product_name}" style="width:100%;height:180px;object-fit:cover;border-radius:8px;">'
    return f'<div style="width:100%;height:180px;background:#ddd;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;color:#666;">{product_name}</div>'


def score_color(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"  # green
    elif score >= 0.6:
        return "#f59e0b"  # amber
    else:
        return "#94a3b8"  # slate


def render_product_card(result: RankedProduct) -> str:
    p = result.product
    pct = int(result.rerank_score * 100)
    color = score_color(result.rerank_score)
    img_html = get_product_image_html(p.image_filename, p.name)
    tags_html = " ".join(
        f'<span style="background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:12px;font-size:11px;">{t}</span>'
        for t in p.tags[:4]
    )
    return f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);transition:box-shadow 0.2s;" onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.12)'" onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.08)'">
      {img_html}
      <div style="padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;">{pct}% match</span>
          <span style="background:#f1f5f9;color:#64748b;padding:2px 8px;border-radius:12px;font-size:11px;">{p.category}</span>
        </div>
        <h3 style="margin:0 0 2px 0;font-size:14px;font-weight:600;color:#1e293b;line-height:1.3;">{p.name}</h3>
        <p style="margin:0 0 4px 0;font-size:12px;color:#64748b;">{p.brand} &nbsp;·&nbsp; <strong style="color:#0f172a;">${p.price:.2f}</strong></p>
        <p style="margin:0 0 8px 0;font-size:12px;color:#475569;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">{p.description}</p>
        <div style="margin-bottom:8px;">{tags_html}</div>
        <div style="border-top:1px solid #f1f5f9;padding-top:8px;">
          <p style="margin:0;font-size:11px;color:#64748b;font-style:italic;line-height:1.4;">💡 {result.match_reason}</p>
        </div>
        <p style="margin:4px 0 0 0;font-size:10px;color:#94a3b8;">Vector: {result.similarity_score:.2f}</p>
      </div>
    </div>
    """


def render_results_grid(results: list[RankedProduct]) -> str:
    if not results:
        return '<p style="color:#64748b;text-align:center;padding:40px;">No results found. Try a different query.</p>'
    cards = "".join(render_product_card(r) for r in results)
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px;padding:4px;">{cards}</div>'


def render_meta_bar(response: SearchResponse) -> str:
    n = len(response.results)
    t = response.search_time_ms
    intent = response.extracted_intent
    category = intent.get("category", "")
    attrs = intent.get("key_attributes", [])
    attrs_str = ", ".join(attrs[:3]) if attrs else ""
    intent_info = f" · Category: <strong>{category}</strong>" if category and category != "unknown" else ""
    intent_info += f" · Attributes: <em>{attrs_str}</em>" if attrs_str else ""
    return f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin-bottom:12px;font-size:13px;color:#475569;">
      <strong>{n} result{"s" if n != 1 else ""}</strong> · {t:.0f}ms{intent_info}
      <br>
      <span style="color:#94a3b8;">Searched as: </span>
      <em style="color:#334155;">"{response.expanded_query}"</em>
    </div>
    """


# ── Search function ───────────────────────────────────────────────────────────

def search(
    query: str,
    uploaded_image: Image.Image | None,
    text_weight: float,
) -> tuple[str, str]:
    """Main search handler called by Gradio."""
    query = (query or "").strip()
    if not query and uploaded_image is None:
        return (
            '<p style="color:#94a3b8;text-align:center;padding:40px;">Enter a search query or upload an image.</p>',
            "",
        )
    if not query:
        query = "product similar to this image"

    image_weight = round(1.0 - text_weight, 2)

    response = pipeline.run(
        query=query,
        image=uploaded_image,
        text_weight=text_weight,
        image_weight=image_weight,
    )

    meta_html = render_meta_bar(response)
    grid_html = render_results_grid(response.results)
    return meta_html + grid_html, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.gr-button-primary { background: #6366f1 !important; border-color: #6366f1 !important; }
.gr-button-primary:hover { background: #4f46e5 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Product Recommendation Agent") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px;">
      <h1 style="font-size:28px;font-weight:700;color:#1e293b;margin:0;">🛍️ Product Recommendation Agent</h1>
      <p style="color:#64748b;margin:6px 0 0;">Multimodal AI-powered product search · CLIP embeddings · GPT-4o reranking</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<h3 style="color:#374151;margin-bottom:8px;">Search</h3>')
            query_input = gr.Textbox(
                placeholder="e.g. comfortable running shoes for flat feet...",
                label="Text Query",
                lines=2,
            )
            image_input = gr.Image(
                label="Upload Image (optional)",
                type="pil",
                sources=["upload", "clipboard"],
                height=200,
            )
            with gr.Row():
                text_weight_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    label="Text weight (1 - value = image weight)",
                    info="Only affects results when image is uploaded",
                )
            search_btn = gr.Button("Search", variant="primary", size="lg")

            gr.HTML("""
            <div style="margin-top:16px;padding:12px;background:#f0fdf4;border-radius:8px;font-size:12px;color:#166534;">
              <strong>How it works:</strong><br>
              1. GPT-4o expands your query &amp; extracts intent<br>
              2. CLIP embeds text + image into a shared vector space<br>
              3. ChromaDB retrieves top-20 nearest products<br>
              4. GPT-4o reranks by semantic relevance &amp; explains matches
            </div>
            """)

        with gr.Column(scale=3):
            gr.HTML('<h3 style="color:#374151;margin-bottom:8px;">Results</h3>')
            error_output = gr.HTML(visible=False)
            results_output = gr.HTML(
                value='<p style="color:#94a3b8;text-align:center;padding:60px;">Results will appear here after your search.</p>'
            )

    search_btn.click(
        fn=search,
        inputs=[query_input, image_input, text_weight_slider],
        outputs=[results_output, error_output],
    )
    query_input.submit(
        fn=search,
        inputs=[query_input, image_input, text_weight_slider],
        outputs=[results_output, error_output],
    )

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 4px;font-size:12px;color:#94a3b8;">
      AIB-RecommendationAgent · Powered by CLIP + ChromaDB + GPT-4o via OpenRouter
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
