"""In-process vLLM backend, built on the official `LLM.chat` batch API.

Requests are submitted as OpenAI-style conversations with `file://` image URLs,
so vLLM applies the model's own chat template. Each image part carries a `uuid`
so vLLM can skip hashing pixels for prefix caching. `LLM.chat` is synchronous;
for concurrent/async serving run `vllm serve` and use the `vllm:` backend.
"""

import asyncio
import os
import resource

from backends import BackendOptions, Request, Result, build_messages

# Defaults tuned to fit long multi-image prompts on a single low-memory GPU.
DEFAULT_ENGINE_ARGS = {
    "max_model_len": 102400,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.9,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "max_num_batched_tokens": 4096,
}


def _image_part(path: str) -> dict:
    path = os.path.abspath(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"file://{path}"},
        "uuid": f"{path}@{os.stat(path).st_mtime_ns}",
    }


class VLLMLocalBackend:
    def __init__(self, model: str, options: BackendOptions):
        import torch
        from vllm import LLM, SamplingParams

        self.name = f"vllm-local:{model}"
        self.options = options
        self._sampling_params = SamplingParams
        engine_args = DEFAULT_ENGINE_ARGS | (options.engine_args or {})
        # Torch's OpenMP thread pool slows down / clashes with engine startup.
        old_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.llm = LLM(
                model=model,
                seed=options.seed,
                allowed_local_media_path=os.getcwd(),
                **engine_args,
            )
        finally:
            torch.set_num_threads(old_num_threads)
        self.chat_template_kwargs = (
            {"enable_thinking": False} if options.reasoning_effort == "none" else None
        )

    def _generate(self, requests: list[Request], on_done) -> list[Result]:
        conversations = [build_messages(r, _image_part) for r in requests]
        params = [
            self._sampling_params(
                temperature=0.0, max_tokens=r.max_tokens, ignore_eos=r.ignore_eos
            )
            for r in requests
        ]
        outputs = self.llm.chat(
            conversations,
            params,
            use_tqdm=False,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        results = []
        for output in outputs:
            results.append(
                Result(
                    text=output.outputs[0].text,
                    input_tokens=len(output.prompt_token_ids or []),
                    cached_tokens=output.num_cached_tokens or 0,
                    output_tokens=len(output.outputs[0].token_ids),
                    truncated=output.outputs[0].finish_reason == "length",
                )
            )
            if on_done:
                on_done()
        return results

    async def generate(self, requests: list[Request], on_done=None) -> list[Result]:
        return await asyncio.to_thread(self._generate, requests, on_done)

    def reset_cache(self) -> None:
        self.llm.reset_prefix_cache()

    def stats(self) -> dict:
        # ru_maxrss is in KiB on Linux, which is where this backend runs in practice.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return {"peak_memory_gib": round(usage / (1 << 20), 1)}

    async def close(self) -> None:
        pass
