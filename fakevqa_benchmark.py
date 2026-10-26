"""Run the FakeVQA benchmark against a vision-language model backend."""

import argparse
import asyncio
import csv
import itertools
import json
import pathlib
import re
import time

from tqdm import tqdm

from backends import BACKENDS, Backend, BackendOptions, Page, Request, Result, make_backend
from generate_benchmark import generate_benchmark_dataset
from ocr import OCR_MODES, text_layers
from questions import QUESTIONS, Question, evaluate_answers, scaling_questions
from scan import SCAN_PRESETS


class Benchmark:
    def __init__(
        self,
        backend: Backend,
        entries: list[dict],
        docs: list[list[Page]],
        output_dir: pathlib.Path,
        max_tokens: int,
    ):
        self.backend = backend
        # Output token cap for quality runs (questions with a larger budget keep theirs)
        self.max_tokens = max_tokens
        # entries and docs are parallel lists, sorted by page count
        self.entries = entries
        self.docs = docs
        self.output_dir = output_dir
        self.metrics: list[dict] = []

    async def run(
        self, tag: str, question: Question, docs: list[list[Page]] | None = None, ignore_eos: bool = False
    ) -> list[Result]:
        """Ask `question` about every document in `docs` (default: all) and record a metrics row."""
        docs = self.docs if docs is None else docs
        prompts = question.prompt if isinstance(question.prompt, list) else [question.prompt]
        # A list of questions shares the token budget, one request per question
        max_tokens = question.max_tokens // len(prompts)
        if not ignore_eos:
            # The question budgets define throughput workloads; elsewhere they are only
            # caps, so be generous to not truncate answers (or thinking).
            max_tokens = max(max_tokens, self.max_tokens)
        requests = [Request(pages, p, max_tokens, ignore_eos) for pages in docs for p in prompts]
        num_pages = sum(len(pages) for pages in docs)
        print(
            f"Processing {len(docs)} documents / {num_pages} pages / {len(requests)} requests "
            f"with {tag} question (max output: {max_tokens} tokens per request) ..."
        )
        start_time = time.perf_counter()
        with tqdm(total=len(requests), desc=tag, leave=False) as pbar:
            results = await self.backend.generate(requests, on_done=lambda: pbar.update(1))
        run_time = time.perf_counter() - start_time
        self.backend.reset_cache()

        input_tokens = sum(r.input_tokens for r in results)
        cached_tokens = sum(r.cached_tokens for r in results)
        output_tokens = sum(r.output_tokens for r in results)
        print(
            f"Time: {run_time:.02f} seconds, per page: {run_time / num_pages:.02f} seconds; "
            f"input tokens: {input_tokens} ({cached_tokens} cached), output tokens: {output_tokens}"
        )
        self.record(tag, docs, run_time, input_tokens, cached_tokens, output_tokens, len(requests))
        return results

    async def run_grouped(self, tag: str, question: Question, ignore_eos: bool = False):
        """Run `question` separately for each group of documents with the same page count."""
        for _, group in itertools.groupby(self.docs, key=len):
            await self.run(tag, question, list(group), ignore_eos)

    def record(self, tag, docs, run_time, input_tokens=0, cached_tokens=0, output_tokens=0, num_requests=0):
        num_pages = sum(len(pages) for pages in docs)
        self.metrics.append(
            {
                "backend": self.backend.name,
                "tag": tag,
                "page_count": round(num_pages / len(docs), 1),
                "num_prompts": len(docs),
                "num_requests": num_requests,
                "total_pages": num_pages,
                "run_time": round(run_time, 2),
                "time_per_page": round(run_time / num_pages, 2),
                "input_tokens": input_tokens,
                "cached_tokens": cached_tokens,
                "output_tokens": output_tokens,
            }
            | self.backend.stats()
        )

    def save_metrics(self, scenario: str, tag: str):
        path = self.output_dir / f"metrics_{scenario}_{tag}.csv"
        fields = list(dict.fromkeys(k for row in self.metrics for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.metrics)
        print(f"Metrics saved to {path}")

    def save_texts(self, name: str, results: list[Result]):
        texts = [
            {"docid": entry["docid"], "title": entry["title"], "text": result.text}
            for entry, result in zip(self.entries, results, strict=True)
        ]
        self.save_json(name, texts)

    def save_json(self, name: str, data):
        with open(self.output_dir / name, "w") as f:
            json.dump(data, f, indent=2)


async def composite_scenario(bench: Benchmark):
    """Throughput with simple questions, then mixed QA (scored), summarization and OCR."""
    await bench.run_grouped("simple", QUESTIONS["simple"])

    results = await bench.run("qa", QUESTIONS["qa"])
    bench.save_json("answers.json", evaluate_answers([r.text for r in results], bench.entries))

    bench.save_texts("summaries.json", await bench.run("summarize", QUESTIONS["summarize"]))
    bench.save_texts("ocr.json", await bench.run("ocr", QUESTIONS["ocr"]))
    bench.save_metrics("composite", "all")


def selected_tags(tag: str) -> list[str]:
    return list(QUESTIONS) if tag == "all" else tag.split(",")


async def grouped_scenario(bench: Benchmark, tag: str):
    """Selected questions, run separately per document size."""
    for t in selected_tags(tag):
        await bench.run_grouped(t, QUESTIONS[t])
    bench.save_metrics("grouped", tag)


async def mixed_scenario(bench: Benchmark, tag: str):
    """Selected questions, run over all document sizes at once."""
    for t in selected_tags(tag):
        results = await bench.run(t, QUESTIONS[t])
        if t == "qa":
            bench.save_json("answers.json", evaluate_answers([r.text for r in results], bench.entries))
    bench.save_metrics("mixed", tag)


async def throughput_scenario(bench: Benchmark, tag: str):
    """Grouped and mixed runs of the selected questions, generating up to max_tokens."""
    for t in selected_tags(tag):
        await bench.run_grouped(t, QUESTIONS[t], ignore_eos=True)
        await bench.run(t, QUESTIONS[t], ignore_eos=True)
    bench.save_metrics("throughput", tag)


async def scaling_scenario(bench: Benchmark):
    """Grouped and mixed runs with 1 to 200 separate questions per document."""
    questions = scaling_questions()
    for t, q in questions.items():
        await bench.run_grouped(t, q, ignore_eos=True)
    for t, q in questions.items():
        await bench.run(t, q, ignore_eos=True)
    bench.save_metrics("scaling", "qa_multi")


async def prepare_documents(entries: list[dict], ocr: str | None) -> tuple[list[list[Page]], float | None]:
    """Turn entries into pages; with `ocr`, attach each page's text layer. Returns (docs, OCR time)."""
    if not ocr:
        return [[Page(f) for f in entry["files"]] for entry in entries], None
    start_time = time.perf_counter()
    num_pages = sum(len(entry["files"]) for entry in entries)
    with tqdm(total=num_pages, desc=f"Text layer ({ocr})") as pbar:
        layers = await text_layers(entries, ocr, on_done=pbar.update)
    run_time = time.perf_counter() - start_time
    docs = [
        [Page(f, text) for f, text in zip(entry["files"], layer, strict=True)]
        for entry, layer in zip(entries, layers, strict=True)
    ]
    return docs, run_time


def parse_engine_args(items: list[str]) -> dict:
    engine_args = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            engine_args[key] = json.loads(value)
        except json.JSONDecodeError:
            engine_args[key] = value
    return engine_args


async def main(args):
    engine_args = {"limit_mm_per_prompt": {"image": args.max_pages}} | parse_engine_args(args.engine_arg)
    options = BackendOptions(
        base_url=args.base_url,
        concurrency=args.concurrency,
        reasoning_effort=None if args.reasoning_effort == "default" else args.reasoning_effort,
        seed=args.seed,
        engine_args=engine_args,
    )
    backend = make_backend(args.model, options)

    run_name = backend.name + (f"+ocr-{args.ocr}" if args.ocr else "") + (f"+scan-{args.scan}" if args.scan else "")
    output_dir = pathlib.Path(args.output_dir) / re.sub(r"[^\w.-]+", "_", run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = generate_benchmark_dataset(
        output_dir=args.data_dir or str(output_dir / "pages"),
        max_pages=args.max_pages,
        min_pages=args.min_pages,
        num_pages=args.total_pages,
        seed=args.seed,
        scan=args.scan,
    )
    # Sort by page count so that documents of the same size can be grouped
    entries.sort(key=lambda e: len(e["files"]))
    docs, ocr_time = await prepare_documents(entries, args.ocr)

    bench = Benchmark(backend, entries, docs, output_dir, args.max_tokens)
    if ocr_time is not None:
        bench.record(f"ocr-{args.ocr}", docs, ocr_time)

    try:
        if args.scenario == "composite":
            await composite_scenario(bench)
        elif args.scenario == "grouped":
            await grouped_scenario(bench, args.tag)
        elif args.scenario == "mixed":
            await mixed_scenario(bench, args.tag)
        elif args.scenario == "throughput":
            await throughput_scenario(bench, args.tag)
        elif args.scenario == "scaling":
            await scaling_scenario(bench)
    finally:
        await backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark vision-language models on synthetic document VQA."
    )
    parser.add_argument(
        "--model",
        "-M",
        required=True,
        help=f"Model as <backend>:<model>, backend one of: {', '.join(BACKENDS)}. "
        "E.g. ollama:qwen3-vl:8b, mistral:mistral-small-latest, vllm-local:Qwen/Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument("--base-url", help="Override the backend's API base URL.")
    parser.add_argument("--concurrency", "-c", type=int, help="Max concurrent requests (HTTP backends).")
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="Reasoning effort to request ('none' disables thinking; 'default' leaves it to the model).",
    )
    parser.add_argument(
        "--ocr",
        help=f"Extract from a text layer instead of page images: {', '.join(OCR_MODES)} "
        "(oracle text from the generator), or mistral[:<ocr model>].",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra vLLM engine argument for vllm-local (value parsed as JSON if possible). Repeatable.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Output token cap per request, unless the question's own budget is larger. "
        "Throughput and scaling runs (ignore_eos) generate exactly the question budgets.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the dataset and model.")
    parser.add_argument(
        "--scan",
        help=f"Make pages look scanned: preset ({', '.join(SCAN_PRESETS)}) and/or key=value "
        "overrides of dpi, oversample, rotation, blur, noise, jpeg_quality; e.g. medium,dpi=150.",
    )
    parser.add_argument("--max-pages", "-m", type=int, default=32, help="Maximum pages per document.")
    parser.add_argument("--min-pages", "-n", type=int, default=4, help="Minimum pages per document.")
    parser.add_argument("--total-pages", "-p", type=int, default=None, help="Total pages in the benchmark.")
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default="simple",
        help=f"Question(s) for grouped/mixed/throughput, comma-separated or 'all': {', '.join(QUESTIONS)}.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=["composite", "grouped", "mixed", "throughput", "scaling"],
        default="composite",
        help="Scenario to run.",
    )
    parser.add_argument("--output-dir", "-o", default="results", help="Directory for metrics and outputs.")
    parser.add_argument(
        "--data-dir", help="Where to generate the page images (default: <output dir>/<run>/pages)."
    )
    asyncio.run(main(parser.parse_args()))
