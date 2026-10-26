"""Question bank and answer scoring for the FakeVQA benchmark."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Question:
    # A single prompt, or a list of prompts sent as separate requests per document
    # (sharing the document prefix); max_tokens is then split evenly between them.
    prompt: str | list[str]
    max_tokens: int


QUESTIONS = {
    "simple": Question("What is the value of the keyword?", 10),
    "qa": Question(
        """Provide the following values based on the document:
    Answer one line per question, just the answer, no other text.
    - docid: document ID
    - title: title of the document
    - keyword: value of the keyword
    - param_b: the value of the parameter B in the table""",
        200,
    ),
    "summarize": Question("Summarize the document in 500 words", 1000),
    "ocr": Question("Extract text from the document", 4000),
    "ocr_long": Question("Extract text from the document", 8000),
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


def qa_expected(entry: dict) -> list[str]:
    """Strings the answer to the `qa` question must contain."""
    return [str(entry[x]) for x in ["docid", "title", "needle"]] + [str(entry["table"][1])]


def score_answer(text: str, entry: dict) -> dict:
    """Score a `qa` answer: how many of the 4 expected values it contains."""
    expected = qa_expected(entry)
    return {"expected": expected, "score": sum(es in text for es in expected)}
