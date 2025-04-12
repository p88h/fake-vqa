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

You can also customize the document sizes:

```bash
python generate_benchmark.py --max_pages 64 --min_pages 8
```

### Running the Benchmark

To run the benchmark with a Vision-Language Model:

```bash
python fakevqa_benchmark.py
```

Options:
- `--seed`: Set the random seed for reproducibility
- `--max-pages` or `-m`: Maximum number of pages to use (choices: 4, 8, 16, 32, 64, 128)
- `--min-pages` or `-n`: Minimum number of pages to use (choices: 1, 2, 4, 8, 16, 32)
- `--total-pages` or `-p`: Total number of pages to use for the benchmark
- `--tag` or `-t`: Type of question to use (choices: simple, qa, summarize, ocr, all)
- `--scenario` or `-s`: Benchmark scenario to run (choices: composite, grouped, mixed, throughput)

#### Benchmark Scenarios

- **Composite**: Runs a comprehensive benchmark with:
  - Throughput test with simple questions
  - Mixed test with QA questions & answer quality assessment 
  - Mixed test with summarization
  - Mixed test with OCR
- **Grouped**: Tests performance on documents grouped by page count
- **Mixed**: Tests performance on a mix of different document sizes
- **Throughput**: Tests both grouped and mixed scenarios for all question types

#### Output Files

The benchmark generates several output files:
- `metrics_{scenario}_{tag}.csv`: Contains performance metrics including:
  - Page count
  - Number of prompts
  - Run time
  - Memory usage
- `answers.json`: Contains the model's answers for QA questions
- `summaries.json`: Contains the model's summaries for summarization tasks
- `ocr.json`: Contains the extracted text from documents

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