"""Text layers for documents: run before extraction when `--ocr` is enabled.

- `passthrough`: the oracle text layer recorded by the generator (exactly what was drawn).
- `mistral[:<model>]`: Mistral OCR over the page images, one PDF per document.
"""

import asyncio
import base64
import io
import os

import httpx
from PIL import Image

OCR_MODES = ["passthrough", "mistral"]
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"


def _pdf_data_url(files: list[str]) -> str:
    images = [Image.open(f).convert("RGB") for f in files]
    buf = io.BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:])
    return "data:application/pdf;base64," + base64.b64encode(buf.getvalue()).decode()


def _page_text(page: dict) -> str:
    parts = [page.get("header"), page["markdown"], page.get("footer")]
    return "\n\n".join(p for p in parts if p)


class MistralOCR:
    def __init__(self, model: str | None, concurrency: int = 4):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set")
        self.model = model or MISTRAL_OCR_MODEL
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
        return [_page_text(p) for p in pages]

    async def close(self) -> None:
        await self.client.aclose()


async def text_layers(entries: list[dict], spec: str, on_done=None) -> list[list[str]]:
    """Return the text of every page of every entry, using the `--ocr` mode in `spec`."""
    mode, _, model = spec.partition(":")
    if mode == "passthrough":
        return [entry["pages_text"] for entry in entries]
    if mode != "mistral":
        raise ValueError(f"Unknown OCR mode {mode!r}; choose from {', '.join(OCR_MODES)}")

    ocr = MistralOCR(model or None)

    async def one(entry):
        pages = await ocr.document(entry["files"])
        if on_done:
            on_done(len(pages))
        return pages

    try:
        return await asyncio.gather(*(one(entry) for entry in entries))
    finally:
        await ocr.close()
