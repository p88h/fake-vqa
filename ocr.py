"""Text layers for documents: run before extraction when `--ocr` is enabled.

- `passthrough`: the oracle text layer recorded by the generator (exactly what was drawn).
- `mistral[:<model>]`: Mistral OCR over the page images, one PDF per document.

Mistral OCR returns figures (our bar charts) as image placeholders, losing the text
inside them. With `figures` (the default), it also annotates each figure with a vision
model, and the placeholder is replaced by the figure's text: a chart's series name and
bar values as a single-row table (`| Units sold | 70 | 74 | 65 |`), like the generated
text layer, plus any other text inside the figure.
"""

import asyncio
import base64
import io
import json
import os

import httpx
from PIL import Image

OCR_MODES = ["passthrough", "mistral"]
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

# Verbatim text only: free-form fields like a caption get filled with descriptions
FIGURE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["series", "values", "text"],
            "properties": {
                "series": {
                    "type": "string",
                    "description": "For a chart: the axis description or series name printed inside it, "
                    "verbatim. Empty if there is none.",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For a bar chart: the value printed on or next to each bar, left to "
                    "right, verbatim. Empty if there are none.",
                },
                "text": {
                    "type": "string",
                    "description": "Any other text printed inside the figure, verbatim, in reading order. "
                    "Empty if there is none. Do not describe the figure.",
                },
            },
        },
    },
}


def _pdf_data_url(files: list[str]) -> str:
    images = [Image.open(f).convert("RGB") for f in files]
    buf = io.BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:])
    return "data:application/pdf;base64," + base64.b64encode(buf.getvalue()).decode()


def _figure_text(annotation: str | None) -> str | None:
    """Render a figure annotation as text; None if there is no usable annotation."""
    try:
        figure = json.loads(annotation or "")
    except json.JSONDecodeError:
        return None
    lines = []
    if figure.get("values"):
        row = ([figure["series"]] if figure.get("series") else []) + figure["values"]
        lines.append("| " + " | ".join(row) + " |")
    elif figure.get("series"):
        lines.append(figure["series"])
    if figure.get("text"):
        lines.append(figure["text"])
    return "\n".join(lines)


def _page_text(page: dict) -> tuple[str, int]:
    """The page's text, with annotated figures in place of their image placeholders.

    Returns the text and the number of figures without a usable annotation.
    """
    markdown, missing = page["markdown"], 0
    for image in page.get("images", []):
        text = _figure_text(image.get("image_annotation"))
        if text is None:
            missing += 1
            continue
        markdown = markdown.replace(f"![{image['id']}]({image['id']})", text)
    parts = [page.get("header"), markdown, page.get("footer")]
    return "\n\n".join(p for p in parts if p), missing


class MistralOCR:
    def __init__(self, model: str | None, figures: bool = True, concurrency: int = 4):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set")
        self.model = model or MISTRAL_OCR_MODEL
        self.figures = figures
        self.missing_figures = 0
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
            transport=httpx.AsyncHTTPTransport(retries=3),
        )
        self.semaphore = asyncio.Semaphore(concurrency)

    async def document(self, files: list[str]) -> list[str]:
        body = {
            "model": self.model,
            "document": {"type": "document_url", "document_url": _pdf_data_url(files)},
        }
        if self.figures:
            body["bbox_annotation_format"] = FIGURE_FORMAT
        async with self.semaphore:
            for attempt in range(6):
                response = await self.client.post(MISTRAL_OCR_URL, json=body)
                if response.status_code != 429 and response.status_code < 500:
                    break
                await asyncio.sleep(2**attempt)
        response.raise_for_status()
        pages = sorted(response.json()["pages"], key=lambda p: p["index"])
        if len(pages) != len(files):
            raise RuntimeError(f"OCR returned {len(pages)} pages for {len(files)} images")
        texts = []
        for page in pages:
            text, missing = _page_text(page)
            texts.append(text)
            self.missing_figures += missing if self.figures else 0
        return texts

    async def close(self) -> None:
        await self.client.aclose()


async def text_layers(
    entries: list[dict], spec: str, figures: bool = True, on_done=None
) -> list[list[str]]:
    """Return the text of every page of every entry, using the `--ocr` mode in `spec`."""
    mode, _, model = spec.partition(":")
    if mode == "passthrough":
        return [entry["pages_text"] for entry in entries]
    if mode != "mistral":
        raise ValueError(f"Unknown OCR mode {mode!r}; choose from {', '.join(OCR_MODES)}")

    ocr = MistralOCR(model or None, figures)

    async def one(entry):
        pages = await ocr.document(entry["files"])
        if on_done:
            on_done(len(pages))
        return pages

    try:
        layers = await asyncio.gather(*(one(entry) for entry in entries))
    finally:
        await ocr.close()
    if ocr.missing_figures:
        print(f"WARNING: {ocr.missing_figures} figures had no usable annotation")
    return layers
