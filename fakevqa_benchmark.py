import resource
import time

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

from vision_language_multi_image import load_qwen2_5_vl
from dataclasses import asdict
from collections import defaultdict
from generate_benchmark import generate_benchmark_dataset


def main(args):
    question = """Answer following questions based on the document:
What is the document ID?
What is the title of the document?
What is the value of the keyword?
What is the values of the parameter B in the table?"""

    entries = generate_benchmark_dataset(max_pages=args.num_pages)
    prompts = []
    req_data = None
    # sort entries by number of pages so that the last entry is the one with the most pages,
    # and we can use the first few for warmup
    entries.sort(key=lambda x: len(x["files"]))
    for entry in entries:
        req_data = load_qwen2_5_vl(question, entry["files"])            
        prompts.append(
            {
                "prompt": req_data.prompt,
                "multi_modal_data": {"image": req_data.image_data},
            }
        )

    # use the engine args from the last entry == one with the most pages 
    # since that will setup correct multimodal limits
    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1024, stop_token_ids=req_data.stop_token_ids
    )
    # first stage - run with groups of equal number of pages
    # Group prompts by number of pages (length of image data)
    grouped_prompts = defaultdict(list)
    for p in prompts:
        grouped_prompts[len(p["multi_modal_data"]["image"])].append(p)
    start_time = time.time()
    for page_count,group in grouped_prompts.items():
        print(f"Running group with {page_count} pages")
        llm.generate(
            group,
            sampling_params=sampling_params,
            lora_request=req_data.lora_requests,
        )
        max_self_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
        max_children_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1 << 20)
        print(f"Peak memory usage: {max_self_usage} (self) + {max_children_usage} (children) GiB")
        print(f"Time: {time.time() - start_time} seconds")
        start_time = time.time()

    # Second time - run a quality check on all the prompts
    # This can perhaps also measure cached performance -- all images are the same
    print("Running the mixed benchmark ...")
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )
    print(f"Time: {time.time() - start_time} seconds")
    total_score = 0
    for i,o in enumerate(outputs):
        generated_text = o.outputs[0].text
        entry = entries[i]
        expected_strings = [ str(entry[x]) for x in ["docid", "title", "needle" ] ] + [ str(entry['table'][1]) ]
        score = 0
        for es in expected_strings:
            if es in generated_text:
                score += 1
        total_score += score
        print(f"Entry {i} output: {generated_text}, expected: {expected_strings}, score: {score * 100 // 4}%")
    print(f"Average score: {total_score * 100 // (len(outputs) * 4)}%")



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--num-pages",
        "-n",
        choices=[4, 8, 16, 32, 64, 128, 256],
        default=32,
        help="Maximum number of page images to use for the benchmark.",
    )

    args = parser.parse_args()
    main(args)
