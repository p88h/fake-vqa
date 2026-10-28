"""Question bank and answer scoring for the FakeVQA benchmark."""

import re
from dataclasses import dataclass
from typing import Callable

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


transcription_prompt = """Transcribe all text in the document exactly as written, page by page, in reading order.
Do not correct, summarize or comment on the text, and do not add an introduction or headings of your own.
Transcribe each bar chart as its title on one line, followed by a single-row table without a header row:
the axis description, then the value shown on each bar from left to right, e.g.
| Units sold | 70 | 74 | 65 |
Output only the transcription."""


@dataclass(frozen=True)
class Question:
    # A single prompt, a function building it from the document entry, or a list of prompts
    # sent as separate requests per document (sharing the document prefix); max_tokens is
    # then split evenly between them.
    prompt: str | list[str] | Callable[[dict], str]
    max_tokens: int


def question_prompts(question: Question, entry: dict) -> list[str]:
    """The prompts to send for `entry`, one request each."""
    if callable(question.prompt):
        return [question.prompt(entry)]
    return question.prompt if isinstance(question.prompt, list) else [question.prompt]


QA_PROMPT = """Provide the following values based on the document:
    Answer one line per question, just the answer, no other text.
    - docid: document ID
    - title: title of the document
    - keyword: value of the keyword
    - param_b: the value of the parameter B in the table"""


def qa_chart(entry: dict) -> dict | None:
    """The chart the `qa` question asks about: the first one whose series name is unique."""
    charts = entry.get("charts", [])
    axes = [c["axis"] for c in charts]
    return next((c for c in charts if axes.count(c["axis"]) == 1), None)


def qa_prompt(entry: dict) -> str:
    # Charts are only drawn where a page has room; documents without one skip the chart field
    chart = qa_chart(entry)
    if not chart:
        return QA_PROMPT
    return QA_PROMPT + f'\n    - chart: the values on the bars of the "{chart["axis"]}" chart, left to right'


QUESTIONS = {
    "simple": Question("What is the value of the keyword?", 10),
    "qa": Question(qa_prompt, 200),
    "summarize": Question("Summarize the document in 500 words", 1000),
    "ocr": Question(transcription_prompt, 4000),
    "ocr_long": Question(transcription_prompt, 8000),
    "qa_batched": Question(qa_system_prompt + "\n".join(qa_batch), 2000),
    "qa_multi": Question([qa_system_prompt + q for q in qa_batch], 2000),
}


def scaling_questions() -> dict[str, Question]:
    """Questions for the scaling scenario: 1 to 200 separate questions per document."""
    questions = {
        "qa_1": Question(qa_batch[0], 200),
        "qa_3": Question(qa_batch[1:4], 600),
        "qa_5": Question(qa_batch[4:9], 1000),
    }
    for s in [1, 2, 3, 5, 10, 20]:
        questions[f"qa_{s}0"] = Question(
            [f"(test case {s}.{d})\n" + qa_batch[d % 10] for d in range(s * 10)],
            s * 2000,
        )
    return questions


def qa_expected(entry: dict) -> dict:
    """Values the answer to the `qa` question must contain, by field."""
    expected = {
        "docid": str(entry["docid"]),
        "title": str(entry["title"]),
        "keyword": str(entry["needle"]),
        "param_b": str(entry["table"][1]),
    }
    if chart := qa_chart(entry):
        expected["chart"] = " ".join(map(str, chart["values"]))
    return expected


def _contains_sequence(text: str, values: str) -> bool:
    """Whether the numbers in `values` appear consecutively among the numbers in `text`."""
    numbers, wanted = re.findall(r"\d+", text), values.split()
    return any(numbers[i : i + len(wanted)] == wanted for i in range(len(numbers) - len(wanted) + 1))


def score_answer(text: str, entry: dict) -> dict:
    """Score a `qa` answer: how many of the expected values it contains."""
    expected = qa_expected(entry)
    found = {
        field: _contains_sequence(text, value) if field == "chart" else value in text
        for field, value in expected.items()
    }
    return {"expected": expected, "found": found, "score": sum(found.values()), "max_score": len(found)}
