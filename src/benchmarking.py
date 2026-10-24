"""The solution registry, the raw calls, the cache in `.cache/`, the runner, parsing and costs."""

import dataclasses
import functools
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from packaging.version import Version
from pydantic import BaseModel
from tqdm.auto import tqdm

from schema import (
    QUESTIONS,
    QUESTIONS_DECISIONS,
    QUESTIONS_SYSTEMONE,
    ROOT,
    Answers,
    DurianAnswer,
    load_prompt,
    prompt_version,
    schema_version,
)

load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "durian_opinions.csv"
PRICES_PATH = ROOT / "config" / "prices.yaml"
CACHE_DIR = ROOT / ".cache"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MIN_VERSION = "0.35.0"
KEV_MIN_VERSION = "0.40.4"
LAYA_MIN_VERSION = "0.40.0"
CLEF_MIN_VERSION = "0.35.1"
OLLAMA_THINK_MIN_VERSION = "0.9.0"
LOCAL_LLM_PREFERENCES = ("qwen3.5:9b", "qwen3.5", "gemma4", "qwen3:8b", "qwen3")
SYSTEM_PROMPT = load_prompt("durian_llm")
PROMPT_VERSION = prompt_version("durian_llm")
SCHEMA_VERSION = schema_version()
LLM_CACHE_VERSION = f"{SCHEMA_VERSION}-prompt{PROMPT_VERSION}"
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}
WARM_UP_QUOTE = "A durian."  # a throwaway text, so the first timed call cannot hit a prompt cache
TIMEOUT_S = 120
GOLD_BINARY_COLUMNS = (
    "gold_smell",
    "gold_taste",
    "gold_texture",
    "gold_value",
    "gold_quality",
    "gold_aftermath",
    "gold_compares",
)
SMOKE_PAIRS = (
    ("musang_king", "en"),
    ("black_thorn", "zh"),
    ("red_prawn", "ms"),
    ("red_prawn", "en"),
    ("black_thorn", "ms"),
)

Family = Literal["decision_cloud", "decision_local", "llm_cloud", "llm_local", "reference"]
RESULT_COLUMNS = (
    "solution",
    "model",
    "opinion_id",
    "cached",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cost_usd",
    "error",
    *(question.name for question in QUESTIONS),
    *(f"{question.name}_confidence" for question in QUESTIONS),
)


@dataclasses.dataclass
class RawCall:
    """One API call as it happened: what was sent (no headers), what came back, how long, usage."""

    request: dict[str, Any]
    response: dict[str, Any]
    latency_s: float
    usage: dict[str, int]


@dataclasses.dataclass
class Solution:
    """One contestant: how to call it on a quote, how to parse its answer, how to check it."""

    id: str
    label: str
    family: Family
    model: str
    price_key: str
    call: Callable[[str], RawCall]
    parse: Callable[[dict[str, Any]], BaseModel]
    confidence: Callable[[dict[str, Any]], dict[str, float]]
    check: Callable[[], str | None]
    optional: bool = False
    cache_version: str = SCHEMA_VERSION

    @property
    def local(self) -> bool:
        return self.family in ("decision_local", "llm_local")


# --- Data ---


def load_opinions(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset with typed columns.

    Args:
        path: The CSV; defaults to `data/durian_opinions.csv` (UTF-8 with BOM).

    Returns:
        One row per opinion: booleans as bool, the 0/1 gold columns as int, `real_score` as float
        (NaN where the writer gave no score), everything else as text ("" when missing).
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in ("names_own_variety", "mentions_other_variety"):
        frame[column] = frame[column] == "True"
    for column in GOLD_BINARY_COLUMNS:
        frame[column] = frame[column].astype(int)
    frame["real_score"] = pd.to_numeric(frame["real_score"].mask(frame["real_score"] == ""))
    return frame


def smoke_subset(opinions: pd.DataFrame) -> pd.DataFrame:
    """
    Five opinions covering all three durians and all three languages, for a quick run.

    Args:
        opinions: The full dataset.

    Returns:
        One opinion per (durian, language) pair in `SMOKE_PAIRS`, preferring one that compares
        durians so the Elo gets a head-to-head match.
    """
    picks = []
    for variety, language in SMOKE_PAIRS:
        candidates = opinions[(opinions["variety"] == variety) & (opinions["language"] == language)]
        comparing = candidates[candidates["gold_compares"] == 1]
        picks.append((comparing if not comparing.empty else candidates).iloc[0])
    return pd.DataFrame(picks).reset_index(drop=True)


# --- Usage helpers ---


def _systemone_usage(raw: dict[str, Any]) -> dict[str, int]:
    usage = raw.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cached_input_tokens": 0,
    }


def _openai_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cached_input_tokens": int(details.get("cached_tokens", 0)),
    }


# --- The raw calls (the notebook shows the same calls inline, on one opinion) ---


def call_jev(quote: str) -> RawCall:
    """One System One request to Jev in the cloud: the quote as `state`, the 18 questions."""
    payload = {"model": "jev-latest", "state": quote, "questions": QUESTIONS_SYSTEMONE}
    headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    start = time.perf_counter()
    response = requests.post(JEV_URL, headers=headers, json=payload, timeout=TIMEOUT_S)
    latency = time.perf_counter() - start
    response.raise_for_status()
    raw = response.json()
    return RawCall(payload, raw, latency, _systemone_usage(raw))


def make_ollama_systemone_call(model: str) -> Callable[[str], RawCall]:
    """The same System One request as Jev's, sent to Ollama's local `/v1/systemone`."""

    def call(quote: str) -> RawCall:
        payload = {"model": model, "state": quote, "questions": QUESTIONS_SYSTEMONE}
        start = time.perf_counter()
        response = requests.post(f"{OLLAMA_HOST}/v1/systemone", json=payload, timeout=TIMEOUT_S)
        latency = time.perf_counter() - start
        response.raise_for_status()
        raw = response.json()
        return RawCall(payload, raw, latency, _systemone_usage(raw))

    return call


@functools.cache
def openai_client() -> Any:
    import openai

    return openai.OpenAI()


def call_openai_decisions(quote: str) -> RawCall:
    """One Decisions API request: the quote as `input`, the 18 questions as a list."""
    request = {"model": "gpt-6-luna", "input": quote, "questions": QUESTIONS_DECISIONS}
    start = time.perf_counter()
    decision = openai_client().decisions.create(**request)
    latency = time.perf_counter() - start
    raw = decision.model_dump(mode="json")
    return RawCall(request, raw, latency, _openai_usage(raw.get("usage")))


def make_openai_llm_call(model: str, effort: str) -> Callable[[str], RawCall]:
    """An ordinary LLM call through the Responses API, parsed into `DurianAnswer`."""

    def call(quote: str) -> RawCall:
        request = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": quote},
            ],
            "reasoning": {"effort": effort},
        }
        start = time.perf_counter()
        response = openai_client().responses.parse(**request, text_format=DurianAnswer)
        latency = time.perf_counter() - start
        answer = response.output_parsed
        if answer is None:
            raise ValueError(f"{model} returned no parsed answer (refusal or incomplete output)")
        raw = {"answer": answer.model_dump(), "response": response.model_dump(mode="json")}
        stored = {**request, "text_format": "DurianAnswer", "prompt_version": PROMPT_VERSION}
        return RawCall(stored, raw, latency, _openai_usage(raw["response"].get("usage")))

    return call


@functools.cache
def ollama_supports_thinking(model: str) -> bool:
    """Whether an Ollama model has a thinking mode (then the call switches it off)."""
    import ollama

    info = ollama.show(model)
    return "thinking" in (getattr(info, "capabilities", None) or [])


def make_ollama_llm_call(model: str) -> Callable[[str], RawCall]:
    """A local LLM through Ollama's chat endpoint with a JSON schema as `format`."""

    def call(quote: str) -> RawCall:
        import ollama

        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": quote},
            ],
            "format": DurianAnswer.model_json_schema(),
            "options": {"temperature": 0},
        }
        if ollama_supports_thinking(model):
            request["think"] = False
        start = time.perf_counter()
        response = ollama.chat(**request)
        latency = time.perf_counter() - start
        answer = DurianAnswer.model_validate_json(response.message.content or "")
        raw = {"answer": answer.model_dump(), "response": response.model_dump(mode="json")}
        usage = {
            "input_tokens": int(response.prompt_eval_count or 0),
            "output_tokens": int(response.eval_count or 0),
            "cached_input_tokens": 0,
        }
        return RawCall({**request, "prompt_version": PROMPT_VERSION}, raw, latency, usage)

    return call


def call_reference(quote: str) -> RawCall:
    """The optional reference: Claude Opus 5.5 at high effort, parsed into `DurianAnswer`."""
    import anthropic

    client = anthropic.Anthropic()
    request = {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": quote}],
        "output_config": {"effort": "high"},
    }
    start = time.perf_counter()
    response = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": quote}],
        output_format=DurianAnswer,
        output_config={"effort": "high"},
    )
    latency = time.perf_counter() - start
    answer = response.parsed_output
    if answer is None:
        raise ValueError(f"{request['model']} returned no parsed answer ({response.stop_reason})")
    raw = {"answer": answer.model_dump(), "response": response.model_dump(mode="json")}
    usage = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "cached_input_tokens": int(response.usage.cache_read_input_tokens or 0),
    }
    return RawCall({**request, "prompt_version": PROMPT_VERSION}, raw, latency, usage)


# --- Parsing into the normalised Answers ---


def score_from_probabilities(probabilities: dict[str, float] | list[float]) -> float:
    """
    Normalise a score answer to 0..1: the probability-weighted level index / (levels - 1).

    Args:
        probabilities: One probability per level, lowest level first - a list, or a dict keyed by
            the legend's index ("0".."n-1" or "1".."n"; any index base works).

    Returns:
        A value in 0..1; 0.0 for a single level.

    Example:
        >>> score_from_probabilities({"0": 0.0, "1": 0.5, "2": 0.5})
        0.75
    """
    if isinstance(probabilities, dict):
        try:
            keys = sorted(probabilities, key=lambda key: int(key))
        except ValueError:
            keys = list(probabilities)
        ordered = [float(probabilities[key]) for key in keys]
    else:
        ordered = [float(value) for value in probabilities]
    if len(ordered) < 2:
        return 0.0
    total = sum(ordered) or 1.0
    weighted = sum(index * value / total for index, value in enumerate(ordered))
    return weighted / (len(ordered) - 1)


def parse_systemone(raw: dict[str, Any]) -> BaseModel:
    """
    Normalise a System One response (Jev, Kev and the other Ollama decision models).

    Args:
        raw: The response JSON with `answers` keyed by question name.

    Returns:
        Yes/no questions as the `noul` probability, choices as the chosen key, scores from the
        level probabilities.
    """
    answers = raw["answers"]
    values: dict[str, float | str] = {}
    for question in QUESTIONS:
        answer = answers[question.name]
        if question.kind == "yes_no":
            values[question.name] = float(answer["noul"])
        elif question.kind == "choice":
            values[question.name] = str(answer["choice"])
        else:
            values[question.name] = score_from_probabilities(answer["probabilities"])
    return Answers(**values)


def parse_openai_decisions(raw: dict[str, Any]) -> BaseModel:
    """
    Normalise a Decisions API response.

    Args:
        raw: The response as a dict with an `answers` list (name, type, probability / choice /
            probabilities).

    Returns:
        Predicates as their probability, choices as the chosen value, scores from the level
        probabilities (ordered by level value).

    Raises:
        ValueError: When the API refused a question.
    """
    answers = {answer["name"]: answer for answer in raw["answers"]}
    values: dict[str, float | str] = {}
    for question in QUESTIONS:
        answer = answers[question.name]
        if answer.get("type") == "refusal":
            raise ValueError(f"the Decisions API refused the question {question.name}")
        if question.kind == "yes_no":
            values[question.name] = float(answer["probability"])
        elif question.kind == "choice":
            values[question.name] = str(answer["choice"])
        else:
            levels = list(answer["probabilities"])
            try:
                levels.sort(key=lambda level: int(level["value"]))
            except ValueError, TypeError:
                pass
            values[question.name] = score_from_probabilities([lvl["probability"] for lvl in levels])
    return Answers(**values)


def parse_llm(raw: dict[str, Any]) -> BaseModel:
    """
    Normalise a structured LLM answer (`DurianAnswer` as a dict under `answer`).

    Args:
        raw: The stored response with the parsed answer under `answer`.

    Returns:
        Booleans as 1.0 / 0.0, levels as their index / (levels - 1), choices as the key.
    """
    answer = DurianAnswer.model_validate(raw["answer"])
    values: dict[str, float | str] = {}
    for question in QUESTIONS:
        value = getattr(answer, question.name)
        if question.kind == "yes_no":
            values[question.name] = 1.0 if value else 0.0
        elif question.kind == "score":
            values[question.name] = question.levels.index(value) / (len(question.levels) - 1)
        else:
            values[question.name] = str(value)
    return Answers(**values)


def _top_probability(probabilities: dict[str, float] | list[float]) -> float:
    values = (
        list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    )
    if not values:
        return 1.0
    total = sum(float(value) for value in values) or 1.0
    return max(float(value) for value in values) / total


def confidence_systemone(raw: dict[str, Any]) -> dict[str, float]:
    """
    The probability of the answer a System One model gave, per question.

    Yes/no: the larger of p and 1 - p. Choice: the chosen option's probability. Score: the most
    likely level's probability.

    Args:
        raw: The response JSON with `answers` keyed by question name.

    Returns:
        Question name -> confidence in 0.5..1 (0..1 for scores).
    """
    answers = raw["answers"]
    confidences: dict[str, float] = {}
    for question in QUESTIONS:
        answer = answers[question.name]
        if question.kind == "yes_no":
            yes = float(answer["noul"])
            confidences[question.name] = max(yes, 1 - yes)
        elif question.kind == "choice":
            probabilities = answer.get("probabilities") or {}
            fallback = float(answer.get("confidence", 1.0))
            confidences[question.name] = float(probabilities.get(answer["choice"], fallback))
        else:
            confidences[question.name] = _top_probability(answer["probabilities"])
    return confidences


def confidence_openai_decisions(raw: dict[str, Any]) -> dict[str, float]:
    """The probability of the answer the Decisions API gave, per question.

    See `confidence_systemone` for the definition per question type.
    """
    answers = {answer["name"]: answer for answer in raw["answers"]}
    confidences: dict[str, float] = {}
    for question in QUESTIONS:
        answer = answers[question.name]
        if question.kind == "yes_no":
            yes = float(answer["probability"])
            confidences[question.name] = max(yes, 1 - yes)
        elif question.kind == "choice":
            chosen = [
                float(option["probability"])
                for option in answer.get("probabilities", [])
                if option["value"] == answer["choice"]
            ]
            confidences[question.name] = (
                chosen[0] if chosen else float(answer.get("confidence", 1.0))
            )
        else:
            levels = [float(level["probability"]) for level in answer["probabilities"]]
            confidences[question.name] = _top_probability(levels)
    return confidences


def confidence_llm(raw: dict[str, Any]) -> dict[str, float]:
    """An LLM's structured answer carries no probability: every question gets confidence 1.0."""
    return {question.name: 1.0 for question in QUESTIONS}


# --- Availability checks ---


def _key_check(name: str) -> Callable[[], str | None]:
    return lambda: None if os.environ.get(name) else f"{name} is not set in .env"


@functools.cache
def ollama_status() -> tuple[str, tuple[str, ...], str | None]:
    """
    Ask the local Ollama server once for its version and the pulled models.

    Returns:
        (version, model tags, error) - the error is a sentence when the server is unreachable.
    """
    try:
        version = requests.get(f"{OLLAMA_HOST}/api/version", timeout=3).json()["version"]
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).json()["models"]
        return version, tuple(model["name"] for model in tags), None
    except (requests.RequestException, KeyError, ValueError) as error:
        return "", (), f"Ollama is not reachable at {OLLAMA_HOST} ({type(error).__name__})"


def _is_pulled(model: str, tags: tuple[str, ...]) -> bool:
    return model in tags or f"{model}:latest" in tags


def _ollama_check(model: str, min_version: str) -> Callable[[], str | None]:
    def check() -> str | None:
        version, tags, error = ollama_status()
        if error:
            return error
        if Version(version) < Version(min_version):
            return f"Ollama {version} is older than {min_version}, needed for {model}"
        if not _is_pulled(model, tags):
            return f"model not pulled (ollama pull {model})"
        return None

    return check


def pick_local_llm() -> str | None:
    """The first pulled Ollama LLM from `LOCAL_LLM_PREFERENCES` (tag or base name), or None."""
    _, tags, _ = ollama_status()
    for preference in LOCAL_LLM_PREFERENCES:
        for tag in tags:
            if tag == preference or tag.split(":")[0] == preference:
                return tag
    return None


@functools.cache
def machine_description() -> str:
    """
    What the local models run on, e.g. "MacBook Air, Apple M2, 24 GB, Ollama 0.40.4".

    Returns:
        Model name, chip and memory from `system_profiler` on a Mac (platform facts elsewhere),
        plus the Ollama version when the server answers.
    """
    parts: list[str] = []
    try:
        report = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        hardware = json.loads(report.stdout)["SPHardwareDataType"][0]
        parts += [hardware.get(key, "") for key in ("machine_name", "chip_type", "physical_memory")]
    except OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError:
        parts += [platform.system(), platform.machine()]
    version, _, _ = ollama_status()
    if version:
        parts.append(f"Ollama {version}")
    return ", ".join(part for part in parts if part)


# --- The registry ---


def registry(local_llm: str | None = None) -> list[Solution]:
    """
    All solutions in display order.

    Args:
        local_llm: The Ollama tag of the local LLM; by default the first pulled one from
            `LOCAL_LLM_PREFERENCES`, falling back to `qwen3.5:9b`.

    Returns:
        The solutions; `availability()` tells which ones can run right now.
    """
    local_llm = local_llm or pick_local_llm() or LOCAL_LLM_PREFERENCES[0]
    kev = "kev:4b-q8_0"
    return [
        Solution(
            "jev", "Jev (TypeSafe)", "decision_cloud", "jev-latest", "jev",
            call_jev, parse_systemone, confidence_systemone, _key_check("TYPESAFE_API_KEY"),
        ),
        Solution(
            "openai_decisions", "OpenAI Decisions (GPT-6 Luna)", "decision_cloud", "gpt-6-luna",
            "openai_decisions", call_openai_decisions, parse_openai_decisions,
            confidence_openai_decisions,
            _key_check("OPENAI_API_KEY"),
        ),
        Solution(
            "kev", "Kev-4B (Ollama)", "decision_local", kev, "ollama_kev",
            make_ollama_systemone_call(kev), parse_systemone, confidence_systemone,
            _ollama_check(kev, KEV_MIN_VERSION),
        ),
        Solution(
            "ollama_d1", "d1 (Liquid AI, Ollama)", "decision_local", "d1", "ollama_d1",
            make_ollama_systemone_call("d1"), parse_systemone,
            confidence_systemone,
            _ollama_check("d1", OLLAMA_MIN_VERSION),
        ),
        Solution(
            "ollama_tev1", "tev1-4B (Together AI, Ollama)", "decision_local", "tev1:4b",
            "ollama_tev1", make_ollama_systemone_call("tev1:4b"), parse_systemone,
            confidence_systemone,
            _ollama_check("tev1:4b", OLLAMA_MIN_VERSION),
        ),
        Solution(
            "ollama_clef_flash", "Clef Flash-9B (Cloudflare, Ollama)", "decision_local",
            "clef-flash", "ollama_clef_flash", make_ollama_systemone_call("clef-flash"),
            parse_systemone, confidence_systemone, _ollama_check("clef-flash", CLEF_MIN_VERSION),
        ),
        Solution(
            "ollama_laya", "Laya (encoder, Ollama)", "decision_local", "laya", "ollama_laya",
            make_ollama_systemone_call("laya"), parse_systemone,
            confidence_systemone,
            _ollama_check("laya", LAYA_MIN_VERSION), optional=True,
        ),
        Solution(
            "ollama_nimble", "Nimble-9B (Bespoke Labs, Ollama)", "decision_local", "nimble",
            "ollama_nimble", make_ollama_systemone_call("nimble"), parse_systemone,
            confidence_systemone,
            _ollama_check("nimble", OLLAMA_MIN_VERSION), optional=True,
        ),
        Solution(
            "gpt_6_luna", "GPT-6 Luna (LLM)", "llm_cloud", "gpt-6-luna", "gpt_6_luna",
            make_openai_llm_call("gpt-6-luna", "none"), parse_llm, confidence_llm,
            _key_check("OPENAI_API_KEY"),
            cache_version=LLM_CACHE_VERSION,
        ),
        Solution(
            "gpt_5_6_terra", "GPT-5.6 Terra (LLM)", "llm_cloud", "gpt-5.6-terra", "gpt_5_6_terra",
            make_openai_llm_call("gpt-5.6-terra", "low"), parse_llm, confidence_llm,
            _key_check("OPENAI_API_KEY"),
            cache_version=LLM_CACHE_VERSION,
        ),
        Solution(
            "ollama_llm", f"{local_llm} (local LLM, Ollama)", "llm_local", local_llm, "ollama_llm",
            make_ollama_llm_call(local_llm), parse_llm,
            confidence_llm,
            _ollama_check(local_llm, OLLAMA_THINK_MIN_VERSION), cache_version=LLM_CACHE_VERSION,
        ),
        Solution(
            "reference", "Claude Opus 5.5 (reference)", "reference", "claude-opus-5-5",
            "reference_claude_opus", call_reference, parse_llm, confidence_llm,
            _key_check("ANTHROPIC_API_KEY"),
            optional=True, cache_version=LLM_CACHE_VERSION,
        ),
    ]  # fmt: skip


def availability(solutions: list[Solution]) -> pd.DataFrame:
    """One row per solution: available, or skipped with the reason."""
    rows = []
    for solution in solutions:
        reason = solution.check()
        rows.append(
            {
                "solution": solution.id,
                "label": solution.label,
                "model": solution.model,
                "status": "available" if reason is None else "skipped",
                "reason": reason or "",
            }
        )
    return pd.DataFrame(rows).set_index("solution")


def available(solutions: list[Solution]) -> list[Solution]:
    """The solutions whose check passes right now."""
    return [solution for solution in solutions if solution.check() is None]


def skipped(solutions: list[Solution]) -> dict[str, str]:
    """Solution id -> reason, for the solutions that cannot run right now."""
    return {solution.id: reason for solution in solutions if (reason := solution.check())}


def solutions_frame(solutions: list[Solution]) -> pd.DataFrame:
    """Label, family, model and `local` per solution id, for the summary table and the charts."""
    rows = [
        {
            "solution": s.id,
            "label": s.label,
            "family": s.family,
            "model": s.model,
            "local": s.local,
        }
        for s in solutions
    ]
    return pd.DataFrame(rows).set_index("solution")


# --- Costs ---


@functools.cache
def load_prices(path: Path = PRICES_PATH) -> dict[str, Any]:
    """Load `config/prices.yaml` (USD per 1M tokens per solution, dated)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cost_usd(price_key: str, usage: dict[str, int]) -> float:
    """
    Price one call from its token usage and the list prices.

    Cached input tokens are billed at `cached_input` where the price list has one; cache writes
    are not priced separately.

    Args:
        price_key: The solution's key in `config/prices.yaml`.
        usage: `input_tokens`, `output_tokens` and optionally `cached_input_tokens`.

    Returns:
        USD for the call; 0.0 for local models.

    Example:
        >>> cost_usd("jev", {"input_tokens": 1_000_000, "output_tokens": 0})
        0.042
    """
    prices = load_prices()["solutions"][price_key]
    cached = usage.get("cached_input_tokens", 0)
    uncached = max(usage.get("input_tokens", 0) - cached, 0)
    cached_price = prices.get("cached_input", prices["input"])
    output = usage.get("output_tokens", 0)
    return (uncached * prices["input"] + cached * cached_price + output * prices["output"]) / 1e6


# --- The cache ---


def cache_path(solution: Solution, opinion_id: str) -> Path:
    """`.cache/<solution>/<model>/<schema_version>/<opinion id>.json`."""
    model_dir = re.sub(r"[^\w.-]+", "_", solution.model)
    return CACHE_DIR / solution.id / model_dir / solution.cache_version / f"{opinion_id}.json"


def read_cache(path: Path) -> dict[str, Any] | None:
    """The cached record, or None when there is none."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(path: Path, record: dict[str, Any]) -> None:
    """Write one record as pretty JSON (UTF-8, no ASCII escapes so the quotes stay readable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


# --- The runner ---


def _retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status in RETRY_STATUSES:
        return True
    if isinstance(error, requests.ConnectionError | requests.Timeout):
        return True
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError"}


def call_with_retries(solution: Solution, quote: str, max_retries: int = 4) -> RawCall:
    """
    Call a solution, retrying with exponential backoff on 429 / 529 / 5xx and connection errors.

    Args:
        solution: The solution to call.
        quote: The opinion text.
        max_retries: Retries after the first attempt; the wait doubles from 2 seconds.

    Returns:
        The raw call.
    """
    for attempt in range(max_retries + 1):
        try:
            return solution.call(quote)
        except Exception as error:
            if attempt == max_retries or not _retryable(error):
                raise
            wait = 2 ** (attempt + 1)
            tqdm.write(f"{solution.id}: {type(error).__name__}, retry {attempt + 1} in {wait} s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _result_row(
    solution: Solution,
    opinion_id: str,
    record: dict[str, Any] | None,
    cached: bool,
    error: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "solution": solution.id,
        "model": solution.model,
        "opinion_id": opinion_id,
        "cached": cached,
        "latency_s": np.nan,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cost_usd": np.nan,
        "error": error,
        **{question.name: None for question in QUESTIONS},
    }
    if record is None:
        return row
    try:
        answers = solution.parse(record["response"])
        confidences = solution.confidence(record["response"])
    except (KeyError, ValueError, TypeError) as parse_error:
        row["error"] = f"parse error: {parse_error}"
        return row
    usage = record["usage"]
    row.update(
        latency_s=record["latency_s"],
        **usage,
        cost_usd=cost_usd(solution.price_key, usage),
        **answers.model_dump(),
        **{f"{name}_confidence": value for name, value in confidences.items()},
    )
    return row


def run_benchmark(
    opinions: pd.DataFrame,
    solutions: list[Solution],
    force_rerun: bool = False,
    max_consecutive_errors: int = 3,
    budget_usd: float | None = None,
) -> pd.DataFrame:
    """
    Run every solution on every opinion, sequentially, through the cache.

    Local models get one untimed warm-up call (on a throwaway text, so the first timed call does
    not hit the server's prompt cache) before their first real one. A solution that fails
    `max_consecutive_errors` times in a row is stopped; its remaining opinions carry that error.

    Args:
        opinions: The opinions to run (the full dataset or `smoke_subset`).
        solutions: The available solutions.
        force_rerun: Ignore cached answers and call the APIs again.
        max_consecutive_errors: When to give up on a solution for this run.
        budget_usd: Stop with a `RuntimeError` once the new (uncached) calls of this run cost
            more than this, at list prices.

    Returns:
        One row per opinion x solution: ids, `cached`, `latency_s`, tokens, `cost_usd`, `error`
        (None when fine) and the 18 normalised answers.
    """
    rows = []
    spent_usd = 0.0
    for solution in solutions:
        warmed_up = False
        consecutive_errors = 0
        stopped: str | None = None
        for opinion in tqdm(list(opinions.itertuples(index=False)), desc=solution.id, leave=False):
            path = cache_path(solution, opinion.id)
            record = None if force_rerun else read_cache(path)
            cached = record is not None
            error = stopped
            if record is None and error is None:
                try:
                    if solution.local and not warmed_up:
                        solution.call(WARM_UP_QUOTE)
                        warmed_up = True
                    raw = call_with_retries(solution, opinion.quote)
                    record = {
                        "solution": solution.id,
                        "model": solution.model,
                        "opinion_id": opinion.id,
                        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                        **dataclasses.asdict(raw),
                    }
                    write_cache(path, record)
                    consecutive_errors = 0
                    spent_usd += cost_usd(solution.price_key, raw.usage)
                    if budget_usd is not None and spent_usd > budget_usd:
                        raise RuntimeError(
                            f"stopped: this run spent USD {spent_usd:.2f}, over the budget of "
                            f"USD {budget_usd:.2f} (answers so far are cached)"
                        )
                except RuntimeError:
                    raise
                except Exception as call_error:
                    error = f"{type(call_error).__name__}: {str(call_error)[:200]}"
                    consecutive_errors += 1
                    tqdm.write(f"{solution.id} / {opinion.id}: {error}")
                    if consecutive_errors >= max_consecutive_errors:
                        stopped = f"stopped after {consecutive_errors} consecutive errors"
            rows.append(_result_row(solution, opinion.id, record, cached, error))
    results = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    for question in QUESTIONS:
        if question.kind != "choice":
            results[question.name] = pd.to_numeric(results[question.name])
        results[f"{question.name}_confidence"] = pd.to_numeric(
            results[f"{question.name}_confidence"]
        )
    results["error"] = results["error"].astype(object).where(results["error"].notna(), None)
    return results
