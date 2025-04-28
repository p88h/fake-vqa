import argparse
import json
import os

from tqdm import tqdm
from generate_mock_doc import generate_mock_document


def generate_benchmark_dataset(
    output_dir="benchmark_docs", max_pages=32, min_pages=4, num_pages=None
):
    """Generate a benchmark dataset with the following structure:
    - 1 x max_pages page document
    - 2 x max_pages/2 page documents
    ...
    - 2^k x min_pages page documents

    Where k = log2(max_pages / min_pages)

    If num_pages is provided, it will be used as the total number of pages to generate.
    The rough distribution of pages per document will remain the same.
    The actual number of pages may be adjusted due to rounding.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the size ratios to use based on the min and max pages.
    ratios = [1]
    while max_pages // ratios[-1] > min_pages:
        ratios.append(ratios[-1] * 2)

    # Computer the total number of pages to generate unless provided
    if num_pages is None:
        num_pages = max_pages * len(ratios)

    # Define the document sets to generate
    document_sets = []
    remaining = len(ratios)
    for r in ratios:
        pages_budget = num_pages // remaining
        pages_per_doc = max_pages // r
        num_docs = pages_budget // pages_per_doc
        document_sets.append(
            {
                "num_docs": num_docs,
                "pages_per_doc": pages_per_doc,
                "prefix": f"doc_{pages_per_doc}p",
            }
        )
        remaining -= 1
        num_pages -= num_docs * pages_per_doc

    total_pages = sum(
        doc_set["num_docs"] * doc_set["pages_per_doc"] for doc_set in document_sets
    )
    # Generate each document set
    entries = []
    with tqdm(total=total_pages, desc="Generating research papers") as pbar:
        for doc_set in document_sets:
            num_docs = doc_set["num_docs"]
            pages_per_doc = doc_set["pages_per_doc"]
            prefix_base = doc_set["prefix"]

            for i in range(num_docs):
                # Create a unique prefix for each document
                prefix = f"{prefix_base}_{i+1:02d}"

                # Generate the document
                result = generate_mock_document(
                    num_pages=pages_per_doc,
                    output_dir=output_dir,
                    prefix=prefix,
                    pbar=pbar,
                )
                entries.append(result)

    with open(os.path.join(output_dir, "benchmark_dataset.json"), "w") as f:
        json.dump(entries, f, indent=2)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate a benchmark dataset for DocVQA"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_docs",
        help="Directory to save the generated documents",
    )
    parser.add_argument(
        "--max_pages", type=int, default=32, help="Maximum number of pages per document"
    )
    parser.add_argument(
        "--min_pages", type=int, default=4, help="Minimum number of pages per document"
    )
    parser.add_argument(
        "--num_pages", type=int, default=None, help="Total number of pages to generate"
    )
    args = parser.parse_args()
    print(
        f"Generating benchmark dataset with {args.min_pages} to {args.max_pages} pages per document"
    )
    _ = generate_benchmark_dataset(
        args.output_dir, args.max_pages, args.min_pages, args.num_pages
    )
    print("Benchmark dataset generation complete!")


if __name__ == "__main__":
    main()
