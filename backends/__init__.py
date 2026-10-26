"""Model backends for the FakeVQA benchmark.

Every backend consumes the same OpenAI-style chat requests: the document pages
first (as images, or as their text layer when OCR is enabled) and the question
last, so that servers with prefix caching can reuse the pages across questions.
"""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class Page:
    path: str
    # Text layer of the page (OCR or oracle). When set, it is sent instead of the image.
    text: str | None = None


@dataclass
class Request:
    pages: list[Page]
    question: str
    max_tokens: int
    # Keep generating up to max_tokens (throughput measurements); only some servers support it.
    ignore_eos: bool = False


@dataclass
class Result:
    text: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    # Generation stopped at max_tokens
    truncated: bool = False


@dataclass
class BackendOptions:
    base_url: str | None = None
    api_key: str | None = None
    concurrency: int | None = None
    # Reasoning effort to request; "none" (the default) disables thinking where the model allows it.
    reasoning_effort: str | None = "none"
    seed: int | None = None
    # Extra keyword arguments for the in-process vLLM engine.
    engine_args: dict | None = None


class Backend(Protocol):
    name: str

    async def generate(
        self, requests: list[Request], on_done: Callable[[], None] | None = None
    ) -> list[Result]: ...

    def reset_cache(self) -> None:
        """Drop any prefix cache so that consecutive runs are measured independently."""

    def stats(self) -> dict:
        """Backend-specific metrics to add to the metrics rows (e.g. peak memory)."""

    async def close(self) -> None: ...


def build_messages(request: Request, image_part: Callable[[str], dict]) -> list[dict]:
    """Build the chat messages for a request, rendering each page image with `image_part`."""
    content = []
    if all(page.text is not None for page in request.pages):
        pages_text = "\n\n".join(
            f"--- Page {i} ---\n{page.text}" for i, page in enumerate(request.pages, 1)
        )
        content.append({"type": "text", "text": pages_text})
    else:
        content.extend(image_part(page.path) for page in request.pages)
    content.append({"type": "text", "text": request.question})
    return [{"role": "user", "content": content}]


BACKENDS = ["ollama", "basert", "vllm", "mistral", "openai", "vllm-local"]


def make_backend(spec: str, options: BackendOptions) -> Backend:
    """Create a backend from a `<backend>:<model>` spec, e.g. `ollama:qwen3-vl:8b`."""
    kind, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"Expected <backend>:<model>, got {spec!r}")
    if kind == "vllm-local":
        from backends.vllm_local import VLLMLocalBackend

        return VLLMLocalBackend(model, options)
    from backends.openai_compat import PRESETS, OpenAICompatBackend

    if kind not in PRESETS:
        raise ValueError(f"Unknown backend {kind!r}; choose from {', '.join(BACKENDS)}")
    return OpenAICompatBackend(kind, model, options)
