import argparse
import asyncio
import itertools
import resource
import time
import json
from tqdm import tqdm
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

from vision_language_multi_image import load_qwen2_5_vl
from dataclasses import asdict
from collections import defaultdict
from generate_benchmark import generate_benchmark_dataset

qa_batch = [
    "What is the value of the keyword?",
    "What is the value of the parameter A in the table?",
    "What is the value of the parameter B in the table?",
    "What is the value of the parameter C in the table?",
    "What is the document ID?",
    "What is the title of the document?",
    "How many pages does the document have?",
    "What is the title of the bar chart on the first page?",
    "What are the value labels of the bar chart on the first page?",
    "What would be a one paragraph summary of the document?",
]

qa_system_prompt = """Answer following questions based on the document.
    Answer one line per question, just the answer, no other text.\n"""

# tag -> (question, max_tokens)
question_bank = {
    "simple": ( "What is the value of the keyword?", 10 ),
    "qa": ( """Provide the following values based on the document:
    Answer one line per question, just the answer, no other text.
    - docid: document ID
    - title: title of the document
    - keyword: value of the keyword
    - param_b: the value of the parameter B in the table""", 200 ),
    "summarize": ( "Summarize the document in 500 words", 1000 ),
    "ocr": ( "Extract text from the document", 4000 ),
    "ocr_long": ( "Extract text from the document", 8000 ),
    "qa_batched": ( qa_system_prompt + "\n".join(qa_batch),  2000),
    "qa_multi": ( [ qa_system_prompt + question for question in qa_batch], 2000 )
}

# prompt wrapper for async processing
async def async_prompt(generator):
    final_output = None
    async for output in generator:
        final_output = output
    return final_output

# sync/async LLM wrapper
class LLMWrapper:
    def __init__(self, req_data):
        engine_args = asdict(req_data.engine_args) | {"seed": args.seed}        
        if args.use_async:
            self.async_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
        else:
            self.llm = LLM(**engine_args)

        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1024, stop_token_ids=req_data.stop_token_ids
        )
        self.lora_requests = req_data.lora_requests
    
    async def process(self, prompts):
        if args.use_async:
            requests = []
            for id, prompt in enumerate(prompts):
                gen = self.async_engine.generate(prompt, self.sampling_params, request_id=str(id), lora_request=self.lora_requests)
                requests.append(async_prompt(gen))
            outputs = await asyncio.gather(*requests)
            await self.async_engine.reset_prefix_cache()
        else:
            outputs = self.llm.generate(prompts, self.sampling_params, lora_request=self.lora_requests)
            self.llm.reset_prefix_cache()
        return outputs

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
    
    return LLMWrapper(req_data), prompts

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

def update_prompt(prompt, question):
    return {
        "prompt": prompt["prompt"].replace("_question_placeholder_", question),
        "multi_modal_data": prompt["multi_modal_data"],
    }

async def process_prompts(wrapper, prompts, tag):
    num_pages = sum(len(p["multi_modal_data"]["image"]) for p in prompts)
    print(f"Processing {len(prompts)} prompts / {num_pages} pages with {tag} question "
          f"(max output: {question_bank[tag][1]} tokens) ...")
    start_time = time.time()
    if isinstance(question_bank[tag][0], list):
        # if the question is a list, we will use it as a batch of questions
        # and generate one output per question
        wrapper.sampling_params.max_tokens = question_bank[tag][1] // len(question_bank[tag][0])
        updated_prompts = [ update_prompt(p, q) for p in prompts for q in question_bank[tag][0] ]
    else:
        wrapper.sampling_params.max_tokens = question_bank[tag][1]
        updated_prompts = [ update_prompt(p, question_bank[tag][0]) for p in prompts ]
    outputs = await wrapper.process(updated_prompts)
    total_memory_usage = report_memory_usage()
    run_time = time.time() - start_time
    time_per_page = run_time / num_pages
    print(f"Time: {run_time:.02f} seconds, per page: {time_per_page:.02f} seconds")
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    for output in outputs:
        input_tokens += len(output.prompt_token_ids) 
        if output.encoder_prompt_token_ids:
            input_tokens += len(output.encoder_prompt_token_ids)
        if output.num_cached_tokens:
            cached_tokens += output.num_cached_tokens
        output_tokens += len(output.outputs[0].token_ids)
    print(f"Total input tokens: {input_tokens}, Cached tokens: {cached_tokens}, Output tokens: {output_tokens}")
    avg_page_count = num_pages / len(prompts)
    metrics = [ f"{tag},{avg_page_count:.01f},{len(prompts)},{num_pages},{run_time:.02f},{time_per_page:.02f},{total_memory_usage:.01f}" ]
    return outputs, metrics

async def process_grouped_by_size(wrapper, prompts, tag):
    start_time = time.time()
    metrics = []
    outputs = []
    wrapper.sampling_params.max_tokens = question_bank[tag][1]
    grouped = itertools.groupby(prompts, key=lambda p: len(p["multi_modal_data"]["image"]))
    for page_count, group in grouped:      
        o, m = await process_prompts(wrapper, list(group), tag)
        outputs.extend(o)
        metrics.extend(m)
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

async def composite_scenario(entries):
    """
    Run a composite scenario with throughput/simple, mixed/qa, and mixed/summarize benchmarks.
    """
    wrapper, prompts = prepare_prompts(entries)

    # Single question, grouped by size, used to measure throughput / memory usage
    _, metrics = await process_grouped_by_size(wrapper, prompts, "simple")

    # QA task
    outputs, metrics1 = await process_prompts(wrapper, prompts, "qa")
    metrics.extend(metrics1)
    answers = evaluate_answers(outputs, entries)
    # save answers to a file for review
    with open("answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    # Summarize task
    outputs, metrics2 = await process_prompts(wrapper, prompts, "summarize")
    metrics.extend(metrics2)
    # save summaries to a file for review
    with open("summaries.json", "w") as f:
        json.dump({entries[i]["title"]: o.outputs[0].text for i, o in enumerate(outputs)}, f, indent=2)

    # OCR task
    outputs, metrics3 = await process_prompts(wrapper, prompts, "ocr")
    metrics.extend(metrics3)
    # save ocr to a file for review
    with open("ocr.json", "w") as f:
        json.dump({entries[i]["title"]: o.outputs[0].text for i, o in enumerate(outputs)}, f, indent=2)

    save_metrics(metrics, "composite", "all")

async def grouped_scenario(entries, tag):
    """
    Run a grouped scenario with selected questions.
    """
    wrapper, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else tag.split(",")
    all_metrics = []
    for t in tags:
        _, metrics = await process_grouped_by_size(wrapper, prompts, t)
        all_metrics.extend(metrics)
    save_metrics(all_metrics, "grouped", tag)

async def mixed_scenario(entries, tag):
    """
    Run a mixed scenario with selected questions.
    """
    wrapper, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else tag.split(",")
    all_metrics = []
    for t in tags:
        _, metrics = await process_prompts(wrapper, prompts, t)
        all_metrics.extend(metrics)
    save_metrics(all_metrics, "mixed", tag)

async def throughput_scenario(entries, tag):
    """
    Run a throughput scenario with selected questions.
    """
    wrapper, prompts = prepare_prompts(entries)
    tags = question_bank.keys() if tag == "all" else tag.split(",")
    all_metrics = []
    wrapper.sampling_params.ignore_eos = True
    for t in tags:
        _, metrics1 = await process_grouped_by_size(wrapper, prompts, t)
        all_metrics.extend(metrics1)
        _, metrics2 = await process_prompts(wrapper, prompts, t)
        all_metrics.extend(metrics2)
    save_metrics(all_metrics, "throughput", tag)

async def main(args):
    entries = generate_benchmark_dataset(max_pages=args.max_pages, 
                                         min_pages=args.min_pages, 
                                         num_pages=args.total_pages)
    if args.scenario == "composite":
        await composite_scenario(entries)
    elif args.scenario == "grouped":
        await grouped_scenario(entries, args.tag)
    elif args.scenario == "mixed":
        await mixed_scenario(entries, args.tag)
    elif args.scenario == "throughput":
        await throughput_scenario(entries, args.tag)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--use-async",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use async processing.",
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
        choices=[2, 4, 8, 16, 32, 64, 128],
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
        type=str,
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
    asyncio.run(main(args))
