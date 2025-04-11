# FakeVQA Benchmark

A benchmark suite for testing Vision-Language Models (VLMs) on document visual question answering tasks.

## Overview

This project provides tools to generate synthetic documents with known content and evaluate Vision-Language Models on their ability to extract specific information from these documents. It's designed to help benchmark the performance of VLMs on document understanding tasks with varying document sizes.

## Features

- Generate synthetic research documents with:
  - Semi-random filler text
  - Known "needle" text (for information retrieval testing)
  - Tables with known values
  - Headers and footers
  - Optional bar charts
- Create benchmark datasets with varying document sizes
- Evaluate VLMs on document understanding tasks
- Measure performance metrics including:
  - Processing time
  - Memory usage
  - Accuracy of information extraction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fakevqa.git
cd fakevqa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating Mock Documents

To generate a single mock document:

```bash
python generate_mock_doc.py
```

This will create a 5-page document in the `mock_docs` directory.

### Generating Benchmark Dataset

To generate a benchmark dataset with the default configuration (256 total pages):

```bash
python generate_benchmark.py
```

This will create:
- 1 x 32 page document
- 2 x 16 page documents
- 4 x 8 page documents
- 8 x 4 page documents

You can specify a custom output directory:

```bash
python generate_benchmark.py --output_dir my_benchmark_data
```

### Running the Benchmark

To run the benchmark with a Vision-Language Model:

```bash
python fakevqa_benchmark.py
```

Options:
- `--seed`: Set the random seed for reproducibility
- `--num-pages` or `-n`: Maximum number of pages to use (choices: 4, 8, 16, 32, 64, 128, 256)

## Project Structure

- `generate_mock_doc.py`: Generates individual mock documents
- `generate_benchmark.py`: Creates benchmark datasets with varying document sizes
- `fakevqa_benchmark.py`: Runs the benchmark on a Vision-Language Model
- `vision_language_multi_image.py`: Helper module for loading VLLM models - copy of the same file from VLLM repository

## Requirements

- Python 3.12+
- Pillow
- Faker
- vLLM (for running the benchmark)

## License

[Apache License 2.0](LICENSE) 