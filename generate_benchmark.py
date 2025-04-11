import argparse
import json
import os
from generate_mock_doc import generate_mock_document

def generate_benchmark_dataset(output_dir="benchmark_docs", max_pages=32):
    """Generate a benchmark dataset with the following structure:
    - 1 x max_pages page document
    - 2 x max_pages/2 page documents
    - 4 x max_pages/4 page documents
    - 8 x max_pages/8 page documents
    Total: max_pages * 4 pages
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the document sets to generate
    document_sets = [
        {"num_docs": d, "pages_per_doc": max_pages//d, "prefix": f"doc_{max_pages//d}p"} for d in [1, 2, 4, 8]
    ]
    
    # Generate each document set
    entries = []
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
                prefix=prefix
            )            
            
            print(f"Generated document: {prefix} with {pages_per_doc} pages")
            entries.append(result)

    with open(os.path.join(output_dir, "benchmark_dataset.json"), "w") as f:
        json.dump(entries, f, indent=2)
    
    return entries

def main():
    parser = argparse.ArgumentParser(description="Generate a benchmark dataset for DocVQA")
    parser.add_argument("--output_dir", type=str, default="benchmark_docs",
                        help="Directory to save the generated documents")
    parser.add_argument("--max_pages", type=int, default=32,
                        help="Maximum number of pages per document")
    args = parser.parse_args()
    print(f"Generating benchmark dataset with {args.max_pages} pages per document")
    entries = generate_benchmark_dataset(args.output_dir, args.max_pages)
    print("Benchmark dataset generation complete!")
    

if __name__ == "__main__":
    main() 