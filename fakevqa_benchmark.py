import resource
import time
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

from vision_language_multi_image import load_qwen2_5_vl
from dataclasses import asdict
from collections import defaultdict
from generate_benchmark import generate_benchmark_dataset

# tag -> (question, max_tokens)
question_bank = {
    "simple": ( "What is the value of the keyword?", 128 ),
    "qa": ( """Answer following questions based on the document:
    Answer one line per question, just the answer, no other text.
    - docid: document ID
    - title: title of the document
    - keyword: value of the keyword
    - param_b: the value of the parameter B in the table""", 1024 ),
    "summarize": ( "Summarize the document in 500 words", 2048 ),
    "ocr": ( "Extract text from the document", 4096 )
}

def update_prompt(prompt, question):
    return {
        "prompt": prompt["prompt"].replace("_question_placeholder_", question),
        "multi_modal_data": prompt["multi_modal_data"],
    }

def prepare_prompts(entries):
    prompts = []
    req_data = None
    # sort entries by number of pages so that the last entry is the one with the most pages,
    # and we can use the first few for warmup
    entries.sort(key=lambda x: len(x["files"]))
    with tqdm(total=len(entries), desc="Preparing prompts") as pbar:
        for entry in entries:
            req_data = load_qwen2_5_vl("_question_placeholder_", entry["files"])
            prompts.append(
                {
                    "prompt": req_data.prompt,
                    "multi_modal_data": {"image": req_data.image_data},
                }
            )
            pbar.update(1)
    
    # use the engine args from the last entry == one with the most pages
    # since that will setup correct multimodal limits
    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1024, stop_token_ids=req_data.stop_token_ids
    )

    return llm, sampling_params, req_data.lora_requests, prompts

def report_memory_usage():
    max_self_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    max_children_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1 << 20)
    total_memory_usage = max_self_usage + max_children_usage
    print(f"Peak memory usage: {max_self_usage:.02f} (self) + {max_children_usage:.02f} (children) = {total_memory_usage:.02f} GiB")    
    return total_memory_usage

def save_metrics(metrics, scenario, tag):
    with open(f"metrics_{scenario}_{tag}.csv", "w") as f:
        f.write("tag,page_count,num_prompts,total_pages,run_time,time_per_page,peak_memory_usage\n")
        f.write("\n".join(metrics))

def grouped_benchmark(llm, sampling_params, lora_requests, prompts, tag):
    # first stage - run with groups of equal number of pages
    # Group prompts by number of pages (length of image data)
    grouped_prompts = defaultdict(list)
    for p in prompts:
        q = update_prompt(p, question_bank[tag][0])
        grouped_prompts[len(p["multi_modal_data"]["image"])].append(q)

    start_time = time.time()
    metrics = []
    outputs = []
    sampling_params.max_tokens = question_bank[tag][1]
    for page_count, group in grouped_prompts.items():
        print(f"Running group with {page_count} pages for {tag} question (max output: "
              f"{question_bank[tag][1]} tokens) ...")
        outputs.extend(llm.generate(
            group,
            sampling_params=sampling_params,
            lora_request=lora_requests,
        ))
        total_memory_usage = report_memory_usage()
        run_time = time.time() - start_time
        num_pages = len(group) * page_count
        time_per_page = run_time / num_pages
        print(f"Time: {run_time:.02f} seconds, per page: {time_per_page:.02f} seconds")
        start_time = time.time()
        metrics.append(f"{tag},{page_count},{len(group)},{num_pages},{run_time:.02f},{time_per_page:.02f},{total_memory_usage:.01f}")
    return outputs, metrics

def mixed_benchmark(llm, sampling_params, lora_requests, prompts, tag):
    # Second time - run a quality check on all the documents with the real question
    # This can perhaps also measure cached performance -- all images are the same
    print(f"Running a mixed benchmark for {tag} question (max output: "
          f"{question_bank[tag][1]} tokens) ...")
    start_time = time.time()
    sampling_params.max_tokens = question_bank[tag][1]
    outputs = llm.generate(
        [update_prompt(p, question_bank[tag][0]) for p in prompts],
        sampling_params=sampling_params,
        lora_request=lora_requests,
    )
    total_memory_usage = report_memory_usage()
    run_time = time.time() - start_time
    num_pages = sum(len(p["multi_modal_data"]["image"]) for p in prompts)
    time_per_page = run_time / num_pages
    print(f"Time: {run_time:.02f} seconds, per page: {time_per_page:.02f} seconds")
    avg_page_count = sum(len(p["multi_modal_data"]["image"]) for p in prompts) / len(prompts)
    metrics = [ f"{tag},{avg_page_count:.01f},{len(prompts)},{num_pages},{run_time:.02f},{time_per_page:.02f},{total_memory_usage:.01f}" ]
    return outputs, metrics

def evaluate_answers(outputs, entries):
    total_score = 0
    score_markers = "😖😕😐😊😍"
    print("Scoring answers: ", end="")
    answers = []
    for i, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        entry = entries[i]
        expected_strings = [str(entry[x]) for x in ["docid", "title", "needle"]] + [
            str(entry["table"][1])
        ]
        score = 0
        for es in expected_strings:
            if es in generated_text:
                score += 1
        total_score += score
        answers.append({"answer": generated_text.split("\n"), "expected": expected_strings})
        print(score_markers[score], end="")
    print(f"\nAverage score: {total_score * 100 // (len(outputs) * 4)}%")
    return answers

def composite_scenario(entries):
    """
    Run a composite scenario with throughput/simple, mixed/qa, and mixed/summarize benchmarks.
    """
    llm, sampling_params, lora_requests, prompts = prepare_prompts(entries)

    # Single question, grouped by size, used to measure throughput / memory usage
    _, metrics = grouped_benchmark(llm, sampling_params, lora_requests, prompts, "simple")

    # QA task
    outputs, metrics1 = mixed_benchmark(llm, sampling_params, lora_requests, prompts, "qa")
    metrics.extend(metrics1)
    answers = evaluate_answers(outputs, entries)
    # save answers to a file for review
    with open("answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    # Summarize task
    outputs, metrics2 = mixed_benchmark(llm, sampling_params, lora_requests, prompts, "summarize")
    metrics.extend(metrics2)
    # save summaries to a file for review
    with open("summaries.json", "w") as f:
        json.dump({entries[i]["title"]: o.outputs[0].text for i, o in enumerate(outputs)}, f, indent=2)

    # OCR task
    outputs, metrics3 = mixed_benchmark(llm, sampling_params, lora_requests, prompts, "ocr")
    metrics.extend(metrics3)
    # save ocr to a file for review
    with open("ocr.json", "w") as f:
        json.dump({entries[i]["title"]: o.outputs[0].text for i, o in enumerate(outputs)}, f, indent=2)

    save_metrics(metrics, "composite", "all")

def grouped_scenario(entries, tag):
    """
    Run a grouped scenario with selected questions.
    """
    llm, sampling_params, lora_requests, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else [tag]
    all_metrics = []
    for t in tags:
        _, metrics = grouped_benchmark(llm, sampling_params, lora_requests, prompts, t)
        all_metrics.extend(metrics)
    save_metrics(all_metrics, "grouped", tag)

def mixed_scenario(entries, tag):
    """
    Run a mixed scenario with selected questions.
    """
    llm, sampling_params, lora_requests, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else [tag]
    all_metrics = []
    for t in tags:
        _, metrics = mixed_benchmark(llm, sampling_params, lora_requests, prompts, t)
        all_metrics.extend(metrics)
    save_metrics(all_metrics, "mixed", tag)

def throughput_scenario(entries, tag):
    """
    Run a throughput scenario with selected questions.
    """
    llm, sampling_params, lora_requests, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else [tag]
    all_metrics = []
    for t in tags:
        _, metrics1 = grouped_benchmark(llm, sampling_params, lora_requests, prompts, t)
        all_metrics.extend(metrics1)
        _, metrics2 = mixed_benchmark(llm, sampling_params, lora_requests, prompts, t)
        all_metrics.extend(metrics2)
    save_metrics(all_metrics, "throughput", tag)

def main(args):
    entries = generate_benchmark_dataset(max_pages=args.max_pages, 
                                         min_pages=args.min_pages, 
                                         num_pages=args.total_pages)
    if args.scenario == "composite":
        composite_scenario(entries)
    elif args.scenario == "grouped":
        grouped_scenario(entries, args.tag)
    elif args.scenario == "mixed":
        mixed_scenario(entries, args.tag)
    elif args.scenario == "throughput":
        throughput_scenario(entries, args.tag)

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
        "--max-pages",
        "-m",
        type=int,
        choices=[4, 8, 16, 32, 64, 128],
        default=32,
        help="Maximum number of page images to use for the benchmark.",
    )
    parser.add_argument(
        "--min-pages",
        "-n",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=4,
        help="Minimum number of page images to use for the benchmark.",
    )
    parser.add_argument(
        "--total-pages",
        "-p",
        type=int,
        default=None,
        help="Total number of page images to use for the benchmark.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        choices=list(question_bank.keys()) + ["all"],
        default="simple",
        help="Type of the question to use in the grouped/mixed scenario.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=["composite", "grouped", "mixed", "throughput"],
        default="composite",
        help="Scenario to run.",
    )
    args = parser.parse_args()
    main(args)
