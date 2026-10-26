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
from questions import QUESTIONS, Question, scaling_questions, score_answer
from scan import SCAN_PRESETS
from transcription import compare, total


# Transcription questions, scored against the generated text layer
TRANSCRIPTION_TAGS = {"ocr", "ocr_long"}


class Benchmark:
    def __init__(
        self,
        backend: Backend,
        entries: list[dict],
        docs: list[list[Page]],
        output_dir: pathlib.Path,
        max_tokens: int,
        ocr: str | None = None,
    ):
        self.backend = backend
        # Output token cap for quality runs (questions with a larger budget keep theirs)
        self.max_tokens = max_tokens
        # entries and docs are parallel lists, sorted by page count
        self.entries = entries
        self.docs = docs
        self.output_dir = output_dir
        # A text layer from an OCR model (not the oracle passthrough) gets scored separately
        self.ocr_model = bool(ocr) and not ocr.startswith("passthrough")
        self.ocr = ocr
        self.metrics: list[dict] = []
        # Per-document outputs and scores of quality runs, by tag
        self.outputs: dict[str, list[dict]] = {}

    async def run(
        self, tag: str, question: Question, indices: list[int] | None = None, ignore_eos: bool = False
    ) -> list[Result]:
        """Ask `question` about the documents at `indices` (default: all) and record a metrics row.

        Quality runs (without ignore_eos) are also scored, where the question has a score.
        """
        indices = list(range(len(self.docs))) if indices is None else indices
        docs = [self.docs[i] for i in indices]
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
        row = self.record(tag, docs, run_time, input_tokens, cached_tokens, output_tokens, len(requests))
        if not ignore_eos:
            row |= self.evaluate(tag, indices, results, len(prompts))
        return results

    async def run_grouped(self, tag: str, question: Question, ignore_eos: bool = False):
        """Run `question` separately for each group of documents with the same page count."""
        for _, group in itertools.groupby(range(len(self.docs)), key=lambda i: len(self.docs[i])):
            await self.run(tag, question, list(group), ignore_eos)

    def evaluate(self, tag: str, indices: list[int], results: list[Result], per_doc: int) -> dict:
        """Keep per-document outputs, scoring them where `tag` has a score; returns overall scores."""
        outputs, qa_scores, edits, copy_edits = [], [], [], []
        for n, i in enumerate(indices):
            entry = self.entries[i]
            doc_results = results[n * per_doc : (n + 1) * per_doc]
            output = {"docid": entry["docid"], "title": entry["title"], "pages": len(entry["files"])}
            if per_doc > 1:
                output["texts"] = [r.text for r in doc_results]
                outputs.append(output)
                continue
            result = doc_results[0]
            output |= {"text": result.text, "truncated": result.truncated}
            if tag == "qa":
                output |= score_answer(result.text, entry)
                qa_scores.append(output["score"])
            elif tag in TRANSCRIPTION_TAGS:
                counts = compare("\n".join(entry["pages_text"]), result.text)
                edits.append(counts)
                output |= counts.rates()
                if self.ocr_model:
                    # How faithfully the model copied the OCR text it was given
                    layer = "\n".join(page.text for page in self.docs[i])
                    copy_counts = compare(layer, result.text)
                    copy_edits.append(copy_counts)
                    output |= copy_counts.rates("copy_")
            outputs.append(output)
        self.outputs.setdefault(tag, []).extend(outputs)

        scores = {}
        if qa_scores:
            score_markers = "😖😕😐😊😍"
            scores["qa_score"] = round(100 * sum(qa_scores) / (4 * len(qa_scores)), 1)
            print(f"QA: {''.join(score_markers[s] for s in qa_scores)} {scores['qa_score']}%")
        if edits:
            scores |= total(edits).rates()
            scores["truncated"] = sum(o["truncated"] for o in outputs)
            summary = f"Transcription: CER {scores['cer']}%, WER {scores['wer']}%"
            if copy_edits:
                scores |= total(copy_edits).rates("copy_")
                summary += f" (copying the OCR text: CER {scores['copy_cer']}%, WER {scores['copy_wer']}%)"
            print(summary + f"; {scores['truncated']} of {len(outputs)} hit the token limit")
        return scores

    def record_text_layer(self, run_time: float):
        """Record the text layer extraction; an OCR model's layer is scored against the generated text."""
        row = self.record(f"ocr-{self.ocr}", self.docs, run_time)
        if not self.ocr_model:
            return
        outputs, edits = [], []
        for entry, pages in zip(self.entries, self.docs, strict=True):
            layer = "\n".join(page.text for page in pages)
            counts = compare("\n".join(entry["pages_text"]), layer)
            edits.append(counts)
            outputs.append(
                {"docid": entry["docid"], "title": entry["title"], "pages": len(pages), "text": layer}
                | counts.rates()
            )
        self.outputs["text-layer"] = outputs
        row |= total(edits).rates()
        print(f"Text layer ({self.ocr}): CER {row['cer']}%, WER {row['wer']}%")

    def record(self, tag, docs, run_time, input_tokens=0, cached_tokens=0, output_tokens=0, num_requests=0) -> dict:
        num_pages = sum(len(pages) for pages in docs)
        row = {
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
        } | self.backend.stats()
        self.metrics.append(row)
        return row

    def save(self, scenario: str, tag: str):
        """Save the metrics, and the per-document outputs of each question."""
        path = self.output_dir / f"metrics_{scenario}_{tag}.csv"
        fields = list(dict.fromkeys(k for row in self.metrics for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.metrics)
        for name, outputs in self.outputs.items():
            with open(self.output_dir / f"outputs_{name}.json", "w") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Metrics and outputs saved to {self.output_dir}")


async def composite_scenario(bench: Benchmark):
    """Simple questions grouped by size, then QA, summarization and OCR over all documents."""
    await bench.run_grouped("simple", QUESTIONS["simple"])
    await bench.run("qa", QUESTIONS["qa"])
    await bench.run("summarize", QUESTIONS["summarize"])
    await bench.run("ocr", QUESTIONS["ocr"])
    bench.save("composite", "all")


def selected_tags(tag: str) -> list[str]:
    return list(QUESTIONS) if tag == "all" else tag.split(",")


async def grouped_scenario(bench: Benchmark, tag: str):
    """Selected questions, run separately per document size."""
    for t in selected_tags(tag):
        await bench.run_grouped(t, QUESTIONS[t])
    bench.save("grouped", tag)


async def mixed_scenario(bench: Benchmark, tag: str):
    """Selected questions, run over all document sizes at once."""
    for t in selected_tags(tag):
        await bench.run(t, QUESTIONS[t])
    bench.save("mixed", tag)


async def throughput_scenario(bench: Benchmark, tag: str):
    """Grouped and mixed runs of the selected questions, generating up to max_tokens."""
    for t in selected_tags(tag):
        await bench.run_grouped(t, QUESTIONS[t], ignore_eos=True)
        await bench.run(t, QUESTIONS[t], ignore_eos=True)
    bench.save("throughput", tag)


async def scaling_scenario(bench: Benchmark):
    """Grouped and mixed runs with 1 to 200 separate questions per document."""
    questions = scaling_questions()
    for t, q in questions.items():
        await bench.run_grouped(t, q, ignore_eos=True)
    for t, q in questions.items():
        await bench.run(t, q, ignore_eos=True)
    bench.save("scaling", "qa_multi")


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

    bench = Benchmark(backend, entries, docs, output_dir, args.max_tokens, args.ocr)
    if ocr_time is not None:
        bench.record_text_layer(ocr_time)

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
