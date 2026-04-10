"""
Image utilities: base64 encode/decode, CLIP preprocessing, placeholder generation.
"""
import base64
import io
import os
from PIL import Image, ImageDraw, ImageFont


CLIP_SIZE = 224  # CLIP input resolution


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64 string (with or without data URL prefix) to a PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def preprocess_for_clip(image: Image.Image) -> Image.Image:
    """Resize and convert image to RGB, ready for CLIP preprocessing."""
    return image.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), Image.LANCZOS)


def load_image(image_path: str) -> Image.Image | None:
    """Load an image from disk, return None if file doesn't exist."""
    if not os.path.exists(image_path):
        return None
    return Image.open(image_path).convert("RGB")


# Color palette for placeholder image backgrounds (one per category)
CATEGORY_COLORS = {
    "footwear": "#4A90D9",
    "outerwear": "#7B68EE",
    "bags": "#E8A838",
    "electronics": "#50C878",
    "home": "#E87070",
}

DEFAULT_COLOR = "#A0A0A0"


def generate_placeholder_image(
    product_name: str,
    category: str = "",
    size: tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Generate a colored placeholder image with product name text.
    Used when real product images aren't available.
    """
    bg_color = CATEGORY_COLORS.get(category.lower(), DEFAULT_COLOR)
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw a lighter inner rectangle for visual depth
    margin = 20
    inner_color = _lighten_hex(bg_color, 0.3)
    draw.rectangle(
        [margin, margin, size[0] - margin, size[1] - margin],
        fill=inner_color,
    )

    # Draw product name, word-wrapped
    words = product_name.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) <= 18:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    text = "\n".join(lines)
    font_size = max(28, 48 - len(product_name) // 2)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Center text
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2

    # Shadow
    draw.multiline_text((x + 2, y + 2), text, fill="#00000040", font=font, align="center")
    # Main text
    draw.multiline_text((x, y), text, fill="white", font=font, align="center")

    return img


def save_placeholder_image(
    product_id: str,
    product_name: str,
    category: str,
    output_dir: str,
) -> str:
    """Generate and save a placeholder image. Returns the filename."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{product_id}.jpg"
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        img = generate_placeholder_image(product_name, category)
        img.save(path, "JPEG", quality=90)
    return filename


def _lighten_hex(hex_color: str, factor: float) -> str:
    """Lighten a hex color by blending toward white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
