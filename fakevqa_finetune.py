import argparse
from itertools import batched
from math import sqrt
import PIL
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import torch

from generate_benchmark import generate_benchmark_dataset
from fakevqa_benchmark import question_bank

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    max_seq_length=128000,
    # use_flash_attention_2=True,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# parse args from command line
parser = argparse.ArgumentParser()
parser.add_argument("--max_pages", "-m", type=int, default=16)
parser.add_argument("--min_pages", "-n", type=int, default=1)
parser.add_argument("--total_pages", "-p", type=int, default=None)
args = parser.parse_args()

# generate dataset
entries = generate_benchmark_dataset(
    max_pages=args.max_pages, min_pages=args.min_pages, num_pages=args.total_pages
)


def convert_to_conversation(entry):
    pages = [
        {"type": "image", "image": PIL.Image.open(file)} for file in entry["files"]
    ]
    expected = [
        "docid: " + entry["docid"],
        "title: " + entry["title"],
        "keyword: " + str(entry["needle"]),
        "param_b: " + str(entry["table"][1]),
    ]
    answers = [{"type": "text", "text": f"value: {val}"} for val in expected]
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question_bank["qa"][0]}] + pages,
        },
        {"role": "assistant", "content": answers},
    ]
    return {"messages": conversation}

def split_text_and_images(message):
    content = message["content"]
    text = []
    images = []
    for item in content:
        if item["type"] == "text":
            text.append(item)
        elif item["type"] == "image":
            text.append({"type": "image"})
            images.append(item["image"])
    return { "role": "user", "content": text }, images 

dataset = [ convert_to_conversation(sample) for sample in entries ]

def prepare_inference_sample(message):
    text, images = split_text_and_images(message)
    input_text = tokenizer.apply_chat_template([ text ], add_generation_prompt = True)
    inputs = tokenizer(
        text=input_text,
        images=images,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    return inputs


# Run a preliminary evaluation to check if the model is working
FastVisionModel.for_inference(model)
control = dataset.pop()
inputs = prepare_inference_sample(control["messages"][0])
print(f"Evaluating control sample on QA task...")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 0.1, do_sample = False, min_p = 0.1)

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)  # Enable for training!

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=128000,
    ),
)

trainer_stats = trainer.train()

FastVisionModel.for_inference(model)
print(f"Validating control sample QA task again...")
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 0.1, do_sample = False, min_p = 0.1)
print("Expected: ", control["messages"][1]["content"])
