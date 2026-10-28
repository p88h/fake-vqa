"""OpenAI chat-completions backend: Ollama, BaseRT, `vllm serve`, Mistral, OpenAI."""

import asyncio
import base64
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import openai

from backends import BackendOptions, Request, Result, build_messages


@dataclass(frozen=True)
class Preset:
    base_url: str
    api_key_env: str | None
    concurrency: int
    # Server accepts `ignore_eos` (vLLM sampling extension).
    ignore_eos: bool = False
    # Extra request fields that disable thinking when reasoning_effort is "none".
    no_thinking: dict = field(default_factory=dict)
    seed_field: str = "seed"
    # Per-request timeout (retried): generous for local servers, short enough to retry hung API calls
    timeout: float = 1800


PRESETS = {
    "ollama": Preset("http://localhost:11434/v1", None, 1),
    # `basert serve` runs one sequence at a time unless started with --continuous-batching.
    "basert": Preset("http://localhost:8453/v1", None, 1, ignore_eos=True),
    "vllm": Preset(
        "http://localhost:8000/v1",
        "VLLM_API_KEY",
        16,
        ignore_eos=True,
        no_thinking={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "mistral": Preset("https://api.mistral.ai/v1", "MISTRAL_API_KEY", 8, seed_field="random_seed", timeout=300),
    "openai": Preset("https://api.openai.com/v1", "OPENAI_API_KEY", 8, timeout=300),
}


def _data_url(path: str, cache: dict[str, str]) -> str:
    if path not in cache:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            cache[path] = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return cache[path]


def _message_text(content) -> str:
    """Message content is a string, or (Mistral reasoning models) a list of chunks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        chunk.get("text", "") for chunk in content if chunk.get("type") == "text"
    )


class OpenAICompatBackend:
    def __init__(self, kind: str, model: str, options: BackendOptions):
        self.preset = PRESETS[kind]
        self.name = f"{kind}:{model}"
        self.model = model
        self.options = options
        self.reasoning_effort = options.reasoning_effort
        api_key = options.api_key or (
            os.environ.get(self.preset.api_key_env) if self.preset.api_key_env else None
        )
        if self.preset.api_key_env and not api_key and kind in ("mistral", "openai"):
            raise RuntimeError(f"{self.preset.api_key_env} is not set")
        self.client = openai.AsyncOpenAI(
            base_url=options.base_url or self.preset.base_url,
            api_key=api_key or "none",
            max_retries=5,
            timeout=self.preset.timeout,
        )
        self.semaphore = asyncio.Semaphore(options.concurrency or self.preset.concurrency)
        self._images: dict[str, str] = {}
        self._warned_ignore_eos = False

    def _image_part(self, path: str) -> dict:
        return {"type": "image_url", "image_url": {"url": _data_url(path, self._images)}}

    def _request_body(self, request: Request) -> dict:
        body = {
            "model": self.model,
            "messages": build_messages(request, self._image_part),
            "max_tokens": request.max_tokens,
            # Greedy decoding; some servers (Mistral) reject temperature 0 unless top_p is 1.
            "temperature": 0.0,
            "top_p": 1.0,
        }
        extra = {}
        if self.options.seed is not None:
            if self.preset.seed_field == "seed":
                body["seed"] = self.options.seed
            else:
                extra[self.preset.seed_field] = self.options.seed
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
            if self.reasoning_effort == "none":
                extra |= self.preset.no_thinking
        if request.ignore_eos:
            if self.preset.ignore_eos:
                extra["ignore_eos"] = True
            elif not self._warned_ignore_eos:
                print(f"WARNING: {self.name} does not support ignore_eos; ignoring it")
                self._warned_ignore_eos = True
        if extra:
            body["extra_body"] = extra
        return body

    async def _complete(self, request: Request):
        body = self._request_body(request)
        try:
            return await self.client.chat.completions.create(**body)
        except openai.BadRequestError as e:
            # e.g. "reasoning_effort 'none' is not supported ...; supported values: ['low', 'high']"
            supported = re.search(r"reasoning_effort.*supported values: \[([^\]]*)\]", str(e))
            sent = body.get("reasoning_effort")
            if not (sent and supported):
                raise
            lowest = supported.group(1).split(",")[0].strip(" '\"")
            if lowest == sent:
                raise
            if self.reasoning_effort == sent:
                print(f"WARNING: {self.name} rejects reasoning_effort={sent!r}, using {lowest!r}")
                self.reasoning_effort = lowest
            return await self.client.chat.completions.create(**self._request_body(request))

    async def _generate_one(self, request: Request, on_done: Callable[[], None] | None) -> Result:
        async with self.semaphore:
            response = await self._complete(request)
        if on_done:
            on_done()
        usage = response.usage
        details = usage.prompt_tokens_details if usage else None
        choice = response.choices[0]
        return Result(
            text=_message_text(choice.message.content),
            input_tokens=usage.prompt_tokens if usage else 0,
            cached_tokens=(details.cached_tokens or 0) if details else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            truncated=choice.finish_reason == "length",
        )

    async def generate(self, requests: list[Request], on_done=None) -> list[Result]:
        return await asyncio.gather(*(self._generate_one(r, on_done) for r in requests))

    def reset_cache(self) -> None:
        pass

    def stats(self) -> dict:
        return {}

    async def close(self) -> None:
        await self.client.close()
