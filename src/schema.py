"""The 18 questions from `config/durian.yaml` and the three payload shapes built from them."""

import dataclasses
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "durian.yaml"
PROMPTS_DIR = ROOT / "prompts"

QuestionKind = Literal["yes_no", "score", "choice"]
YAML_TYPE_TO_KIND: dict[str, QuestionKind] = {
    "yes_no": "yes_no",
    "score": "score",
    "choice": "choice",
}


@dataclasses.dataclass(frozen=True)
class Question:
    """One question asked about every opinion, in a format-independent form."""

    name: str
    kind: QuestionKind
    instructions: str
    gold: bool
    levels: tuple[str, ...] = ()
    options: dict[str, str] = dataclasses.field(default_factory=dict)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load the question config as a dict.

    Args:
        path: The YAML file; defaults to `config/durian.yaml`.

    Returns:
        The parsed YAML (version, durians, aspects, question_templates, questions).
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fill_durian_names(text: str, durians: dict[str, str]) -> str:
    return re.sub(r"\{durians\.(\w+)\}", lambda match: durians[match.group(1)], text)


def load_questions(path: Path = CONFIG_PATH) -> list[Question]:
    """
    Expand the config into the 18 questions, in the order every API receives them.

    The six aspects give one `<aspect>_mentioned` yes/no question and one `<aspect>_sentiment`
    score question each; the six standalone questions follow.

    Args:
        path: The YAML file; defaults to `config/durian.yaml`.

    Returns:
        The questions: 6 mentioned, 6 sentiment, then overall_rating, overall_sentiment, humour,
        compares, preferred and variety.

    Example:
        >>> [q.name for q in load_questions()][:2]
        ['smell_mentioned', 'taste_mentioned']
    """
    config = load_config(path)
    templates = config["question_templates"]
    questions: list[Question] = []
    for template_name, kind_name in (("mentioned", "yes_no"), ("sentiment", "score")):
        template = templates[template_name]
        for aspect, definition in config["aspects"].items():
            questions.append(
                Question(
                    name=f"{aspect}_{template_name}",
                    kind=YAML_TYPE_TO_KIND[kind_name],
                    instructions=template["instructions"].format(aspect_definition=definition),
                    gold=template["gold"],
                    levels=tuple(template.get("levels", ())),
                )
            )
    for name, spec in config["questions"].items():
        options = {
            key: _fill_durian_names(description, config["durians"])
            for key, description in spec.get("options", {}).items()
        }
        questions.append(
            Question(
                name=name,
                kind=YAML_TYPE_TO_KIND[spec["type"]],
                instructions=spec["instructions"].strip(),
                gold=spec["gold"],
                levels=tuple(spec.get("levels", ())),
                options=options,
            )
        )
    return questions


def schema_version(path: Path = CONFIG_PATH) -> str:
    """Return the config version as a cache folder name, e.g. `v3`."""
    return f"v{load_config(path)['version']}"


def to_systemone(questions: list[Question]) -> dict[str, dict[str, Any]]:
    """
    Build the `questions` object of a System One request (Jev, and Ollama's `/v1/systemone`).

    Args:
        questions: The questions from `load_questions()`.

    Returns:
        A dict keyed by question name: `noul` questions carry only `instructions`, `choice`
        questions a `criteria` dict (option -> description), `score` questions a `criteria` list
        of levels, lowest first.
    """
    payload: dict[str, dict[str, Any]] = {}
    for question in questions:
        if question.kind == "yes_no":
            payload[question.name] = {"type": "noul", "instructions": question.instructions}
        elif question.kind == "choice":
            payload[question.name] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.options),
            }
        else:
            payload[question.name] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.levels),
            }
    return payload


def to_openai_decisions(questions: list[Question]) -> list[dict[str, Any]]:
    """
    Build the `questions` list of an OpenAI Decisions API request.

    Args:
        questions: The questions from `load_questions()`.

    Returns:
        A list of question dicts: `predicate` (yes/no), `choice` with `choices: [{value,
        description}]`, and `score` with `levels: [{label, description}]`, lowest first.
    """
    payload: list[dict[str, Any]] = []
    for question in questions:
        if question.kind == "yes_no":
            payload.append(
                {"type": "predicate", "name": question.name, "instructions": question.instructions}
            )
        elif question.kind == "choice":
            payload.append(
                {
                    "type": "choice",
                    "name": question.name,
                    "instructions": question.instructions,
                    "choices": [
                        {"value": key, "description": description}
                        for key, description in question.options.items()
                    ],
                }
            )
        else:
            payload.append(
                {
                    "type": "score",
                    "name": question.name,
                    "instructions": question.instructions,
                    "levels": [{"label": level, "description": level} for level in question.levels],
                }
            )
    return payload


def _literal(values: tuple[str, ...]) -> Any:
    """A `Literal[...]` of the given strings, built at runtime."""
    return Literal[values]  # ty: ignore[invalid-type-form]


def answer_model(questions: list[Question]) -> type[BaseModel]:
    """
    Build the pydantic model an LLM fills in (structured output), one field per question.

    Yes/no questions become `bool`, score questions a `Literal` of their level labels, choice
    questions a `Literal` of their option keys. The field description carries the question.

    Args:
        questions: The questions from `load_questions()`.

    Returns:
        A model class named `DurianAnswer` with the 18 fields in question order.
    """
    fields: dict[str, Any] = {}
    for question in questions:
        if question.kind == "yes_no":
            fields[question.name] = (bool, Field(description=question.instructions))
        elif question.kind == "score":
            fields[question.name] = (
                _literal(question.levels),
                Field(description=f"{question.instructions} Levels from lowest to highest."),
            )
        else:
            options = "; ".join(f"{key}: {text}" for key, text in question.options.items())
            fields[question.name] = (
                _literal(tuple(question.options)),
                Field(description=f"{question.instructions} Options: {options}."),
            )
    return create_model("DurianAnswer", __config__=ConfigDict(extra="forbid"), **fields)


def answers_model(questions: list[Question]) -> type[BaseModel]:
    """
    Build the normalised `Answers` model every solution is parsed into.

    Yes/no and score questions become floats in 0..1 (a probability for decision models, 1.0 or
    0.0 for LLMs; scores are the probability-weighted level index divided by levels - 1); choice
    questions keep the chosen option key.

    Args:
        questions: The questions from `load_questions()`.

    Returns:
        A model class named `Answers` with the 18 fields in question order.
    """
    fields: dict[str, Any] = {
        question.name: (str, ...) if question.kind == "choice" else (float, ...)
        for question in questions
    }
    return create_model("Answers", **fields)


def load_prompt(name: str) -> str:
    """
    Load a prompt from `prompts/<name>.md` without its `PROMPT_VERSION` header line.

    Args:
        name: File name without extension, e.g. "durian_llm".

    Returns:
        The prompt text.
    """
    header, _, body = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").partition("\n")
    if not header.startswith("PROMPT_VERSION:"):
        raise ValueError(f"prompts/{name}.md must start with a PROMPT_VERSION line")
    return body.strip()


def prompt_version(name: str) -> str:
    """Return the `PROMPT_VERSION` value of `prompts/<name>.md`, e.g. "1.0"."""
    header = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").partition("\n")[0]
    return header.removeprefix("PROMPT_VERSION:").strip()


QUESTIONS = load_questions()
QUESTIONS_SYSTEMONE = to_systemone(QUESTIONS)
QUESTIONS_DECISIONS = to_openai_decisions(QUESTIONS)
DurianAnswer = answer_model(QUESTIONS)
Answers = answers_model(QUESTIONS)
