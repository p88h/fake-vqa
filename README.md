# FakeVQA Benchmark

A benchmark suite for testing Vision-Language Models (VLMs) on document visual question answering tasks.

## Overview

This project generates synthetic documents with known content and evaluates Vision-Language Models on their ability to extract specific information from them. It's designed to benchmark VLMs on document understanding tasks with varying document sizes, across local and hosted model backends.

## Features

- Generate synthetic research documents with:
  - Semi-random filler text
  - Known "needle" text (for information retrieval testing)
  - Tables with known values
  - Headers and footers
  - Optional bar charts
  - A ground-truth text layer (exactly what was drawn on each page)
- Create benchmark datasets with varying document sizes
- Evaluate VLMs on document understanding tasks, either from page images or from a text layer (OCR or oracle)
- Measure run time, time per page, input / cached / output tokens, and answer accuracy

## Installation

```bash
git clone https://github.com/yourusername/fakevqa.git
cd fakevqa
uv sync                  # HTTP backends (Ollama, BaseRT, vllm serve, Mistral, OpenAI)
uv sync --extra vllm     # + in-process vLLM backend
uv sync --extra finetune # + fakevqa_finetune.py dependencies
```

## Usage

### Generating Mock Documents

To generate a single mock document:

```bash
uv run generate_mock_doc.py
```

This will create a 5-page document in the `mock_docs` directory.

### Generating Benchmark Dataset

To generate a benchmark dataset with the default configuration (128 total pages):

```bash
uv run generate_benchmark.py
```

This will create:
- 1 x 32 page document
- 2 x 16 page documents
- 4 x 8 page documents
- 8 x 4 page documents

Options: `--output_dir`, `--max_pages`, `--min_pages`, `--num_pages` (total pages) and `--seed` (reproducible dataset).

### Running the Benchmark

The benchmark generates a fresh dataset and runs a scenario against one model, given as `<backend>:<model>`:

```bash
uv run fakevqa_benchmark.py -M ollama:gemma4:26b
uv run fakevqa_benchmark.py -M mistral:mistral-small-latest -s mixed -t qa
uv run fakevqa_benchmark.py -M mistral:mistral-small-latest --ocr mistral
uv run fakevqa_benchmark.py -M vllm-local:Qwen/Qwen3-VL-8B-Instruct -s throughput -t all
```

#### Backends

| Backend | Default endpoint | Notes |
|---|---|---|
| `ollama` | `http://localhost:11434/v1` | Ollama's context window (32k by default here) silently truncates long documents; raise `OLLAMA_CONTEXT_LENGTH` for large page counts. Thinking-only models (e.g. `qwen3-vl:8b`) cannot disable reasoning; use an `-instruct` tag. |
| `basert` | `http://localhost:8453/v1` | `basert serve <model>`. v0.3.0 accepts only one image per request: use 1-page documents (`-m 1 -n 1`) or `--ocr`. Serves one request at a time unless started with `--continuous-batching`. |
| `vllm` | `http://localhost:8000/v1` | `vllm serve <model> --limit-mm-per-prompt '{"image": 32}'`. Supports `ignore_eos`. |
| `mistral` | `https://api.mistral.ai/v1` | Needs `MISTRAL_API_KEY`. E.g. `mistral-small-latest`, `mistral-large-4`, `zai-glm-5-3-flash`. |
| `openai` | `https://api.openai.com/v1` | Any other OpenAI-compatible server via `--base-url`. Uses `OPENAI_API_KEY`. |
| `vllm-local` | in-process | vLLM's `LLM.chat` batch API (needs `--extra vllm`, in practice a CUDA GPU). Engine options via `--engine-arg KEY=VALUE`. Reports peak memory. |

All HTTP backends receive the same OpenAI chat request: page images (base64 data URLs) first, the question last, so servers with prefix caching can reuse the pages across questions.

#### Options

- `--model` or `-M`: Model as `<backend>:<model>` (required)
- `--base-url`: Override the backend's endpoint
- `--concurrency` or `-c`: Max concurrent requests (defaults per backend)
- `--reasoning-effort`: Defaults to `none` (thinking off, this is extraction). Models that reject `none` fall back to their lowest supported effort. `default` leaves it to the model.
- `--ocr`: Extract from a text layer instead of page images:
  - `passthrough`: the oracle text layer recorded by the generator
  - `mistral[:<ocr model>]`: Mistral OCR (`mistral-ocr-latest` by default), one PDF per document
- `--seed`: Random seed for the dataset (and the model, where supported)
- `--max-pages` or `-m`, `--min-pages` or `-n`, `--total-pages` or `-p`: Dataset shape
- `--tag` or `-t`: Question(s) for grouped/mixed/throughput, comma-separated or `all`: `simple`, `qa`, `summarize`, `ocr`, `ocr_long`, `qa_batched`, `qa_multi`
- `--scenario` or `-s`: `composite`, `grouped`, `mixed`, `throughput`, `scaling`
- `--output-dir` or `-o`: Where results go (default `results/`)

#### Benchmark Scenarios

- **Composite**: Runs a comprehensive benchmark with:
  - Simple questions grouped by document size
  - Mixed test with QA questions & answer quality assessment
  - Mixed test with summarization
  - Mixed test with OCR
- **Grouped**: Tests performance on documents grouped by page count
- **Mixed**: Tests performance on a mix of different document sizes (scores `qa` answers)
- **Throughput**: Grouped and mixed runs generating up to the token limit (`ignore_eos`, where supported)
- **Scaling**: 1 to 200 separate questions per document, sharing the document prefix

#### Output Files

Results go to `results/<backend>_<model>[_ocr-<mode>]/`:
- `metrics_{scenario}_{tag}.csv`: Per run: backend, question tag, average page count, number of documents and requests, total pages, run time, time per page, input / cached / output tokens (and peak memory for `vllm-local`). With `--ocr`, the first row is the text layer extraction time.
- `answers.json`: The model's answers for QA questions, with the expected values and score
- `summaries.json`: The model's summaries for summarization tasks
- `ocr.json`: The extracted text from documents

## Project Structure

- `generate_mock_doc.py`: Generates individual mock documents and their text layer
- `generate_benchmark.py`: Creates benchmark datasets with varying document sizes
- `fakevqa_benchmark.py`: Runs the benchmark scenarios against a backend
- `questions.py`: Question bank and answer scoring
- `ocr.py`: Text layers for `--ocr` (oracle passthrough, Mistral OCR)
- `backends/`: Model backends (`openai_compat.py` for all HTTP APIs, `vllm_local.py` for in-process vLLM)
- `fakevqa_finetune.py`: LoRA fine-tuning of Qwen2-VL on the QA task (Unsloth)

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## License

[Apache License 2.0](LICENSE)
