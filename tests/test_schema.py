"""The 18 questions and the three payload shapes built from config/durian.yaml."""

import pytest

from schema import (
    QUESTIONS,
    Answers,
    DurianAnswer,
    answer_model,
    load_prompt,
    prompt_version,
    schema_version,
    to_openai_decisions,
    to_systemone,
)

ASPECTS = ("smell", "taste", "texture", "value", "quality", "aftermath")


def test_eighteen_questions_in_order() -> None:
    names = [question.name for question in QUESTIONS]
    assert len(names) == 18
    assert names[:6] == [f"{aspect}_mentioned" for aspect in ASPECTS]
    assert names[6:12] == [f"{aspect}_sentiment" for aspect in ASPECTS]
    assert names[12:] == [
        "overall_rating",
        "overall_sentiment",
        "humour",
        "compares",
        "preferred",
        "variety",
    ]


def test_level_counts_and_option_keys() -> None:
    by_name = {question.name: question for question in QUESTIONS}
    assert len(by_name["taste_sentiment"].levels) == 5
    assert len(by_name["overall_rating"].levels) == 10
    assert len(by_name["humour"].levels) == 4
    assert list(by_name["overall_sentiment"].options) == [
        "positive",
        "mixed",
        "negative",
        "neutral",
    ]
    assert list(by_name["variety"].options) == [
        "musang_king",
        "black_thorn",
        "red_prawn",
        "not_named",
    ]
    assert by_name["variety"].gold and not by_name["humour"].gold


def test_placeholders_are_filled() -> None:
    for question in QUESTIONS:
        assert "{" not in question.instructions
        assert all("{" not in text for text in question.options.values())
    taste = next(question for question in QUESTIONS if question.name == "taste_mentioned")
    assert "sweetness" in taste.instructions
    preferred = next(question for question in QUESTIONS if question.name == "preferred")
    assert preferred.options["musang_king"].startswith("Musang King")


def test_systemone_payload_shapes() -> None:
    payload = to_systemone(QUESTIONS)
    assert len(payload) == 18
    assert payload["smell_mentioned"] == {
        "type": "noul",
        "instructions": payload["smell_mentioned"]["instructions"],
    }
    assert payload["overall_sentiment"]["type"] == "choice"
    assert isinstance(payload["overall_sentiment"]["criteria"], dict)
    assert payload["overall_rating"]["type"] == "score"
    assert payload["overall_rating"]["criteria"][0].startswith("1")
    assert payload["overall_rating"]["criteria"][-1].startswith("10")


def test_openai_decisions_payload_shapes() -> None:
    payload = to_openai_decisions(QUESTIONS)
    assert [question["name"] for question in payload] == [q.name for q in QUESTIONS]
    predicate = payload[0]
    assert predicate["type"] == "predicate" and "instructions" in predicate
    choice = next(question for question in payload if question["name"] == "variety")
    assert [option["value"] for option in choice["choices"]][:3] == [
        "musang_king",
        "black_thorn",
        "red_prawn",
    ]
    score = next(question for question in payload if question["name"] == "humour")
    assert [level["label"] for level in score["levels"]] == [
        "not humorous",
        "slightly humorous",
        "clearly humorous",
        "laugh out loud",
    ]


def test_answer_model_is_strict_with_one_field_per_question() -> None:
    schema = DurianAnswer.model_json_schema()
    assert len(schema["properties"]) == 18
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["compares"]["type"] == "boolean"
    assert schema["properties"]["humour"]["enum"][0] == "not humorous"
    assert schema["properties"]["preferred"]["enum"][-1] == "none"
    assert "Options:" in schema["properties"]["overall_sentiment"]["description"]
    assert answer_model(QUESTIONS).model_fields.keys() == DurianAnswer.model_fields.keys()


def test_answers_model_types() -> None:
    fields = Answers.model_fields
    assert fields["smell_mentioned"].annotation is float
    assert fields["overall_rating"].annotation is float
    assert fields["variety"].annotation is str
    with pytest.raises(ValueError):
        Answers(**{name: "x" for name in fields})


def test_prompt_loader() -> None:
    assert prompt_version("durian_llm") == "1.0"
    assert load_prompt("durian_llm").startswith("You read one short opinion")
    assert "PROMPT_VERSION" not in load_prompt("durian_llm")
    assert schema_version() == "v3"
