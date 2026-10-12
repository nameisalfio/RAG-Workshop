"""The evaluation must score retrieval and answers as the definitions say, and refuse a broken golden set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from src.config import load_config, project_path
from src.evaluation import (
    ContextItem,
    GoldenQuestion,
    check_answer,
    judge_faithfulness,
    load_golden_set,
    retrieval_result,
)
from src.ingestion.chunking import Chunk
from src.llm import ChatClient
from src.retrieval.vector_store import Hit

CONFIG = load_config()
QUESTION = GoldenQuestion(
    id="notice",
    type="superseded",
    question="Con quanto preavviso si recede?",
    answer="Con 45 giorni, per effetto dell'addendum.",
    facts=(("45 giorni", "quarantacinque giorni"), ("addendum",)),
    context=(ContextItem("CTR-017_addendum-01", "Art. 1"), ContextItem("EML-04", "")),
)
NO_ANSWER = GoldenQuestion("phone", "unanswerable", "Qual è il numero?", "Non è nei documenti.", (), ())


def hit(document: str, section: str, text: str = "") -> Hit:
    """A retrieved chunk of a document."""
    return Hit(Chunk(f"folder/{document}", 0, text, section, 1), 0.5)


def test_shipped_golden_set_is_valid() -> None:
    questions = load_golden_set(project_path(CONFIG.paths.golden_set))
    assert len(questions) == 20
    assert sum(not question.answerable for question in questions) == 5
    assert {question.type for question in questions} >= {"direct", "superseded", "lookalike", "unanswerable"}
    assert all(question.facts and question.context for question in questions if question.answerable)
    assert all(not question.facts for question in questions if not question.answerable)


def test_broken_golden_sets_are_errors(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    entry = "- {id: a, type: direct, question: Q, answer: A, facts: [[x]], context: [{document: D}]}\n"
    path.write_text(entry + entry, encoding="utf-8")
    with pytest.raises(ValueError, match="same id"):
        load_golden_set(path)
    incomplete = "- {id: a, type: direct, question: Q, answer: A, facts: [], context: []}\n"
    path.write_text(incomplete, encoding="utf-8")
    with pytest.raises(ValueError, match="must have facts and context"):
        load_golden_set(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_golden_set(path)


def test_retrieval_scores_follow_the_definitions() -> None:
    hits = [
        hit("CTR-017_Beta.pdf", "Art. 7", "preavviso di 60 giorni"),
        hit("CTR-017_addendum-01.pdf", "Art. 1 - Modifica", "preavviso di 45 giorni"),
        hit("CTR-017_addendum-01.pdf", "premesso che", "le Parti intendono"),
        hit("EML-04_chiarimento.txt", "", "per effetto dell'addendum n. 1"),
    ]
    result = retrieval_result(QUESTION, hits)
    assert result.relevant == (False, True, False, True)
    assert result.precision == pytest.approx(2 / 4)
    assert result.recall == pytest.approx(1.0)
    assert result.reciprocal_rank == pytest.approx(1 / 2)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert result.ndcg == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert result.context_recall == pytest.approx(1.0)


def test_retrieval_that_misses_scores_zero_and_cannot_score_what_has_no_answer() -> None:
    missed = retrieval_result(QUESTION, [hit("MM-P7.pdf", "E-4012", "infiltrazione d'acqua")])
    assert (missed.precision, missed.recall, missed.reciprocal_rank, missed.ndcg) == (0.0, 0.0, 0.0, 0.0)
    assert missed.context_recall == 0.0
    half = retrieval_result(QUESTION, [hit("EML-04_chiarimento.txt", "", "sono 45 giorni")])
    assert half.recall == pytest.approx(0.5)
    assert half.context_recall == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unanswerable"):
        retrieval_result(NO_ANSWER, [hit("MM-P7.pdf", "E-4012")])
    with pytest.raises(ValueError, match="No chunks"):
        retrieval_result(QUESTION, [])


def test_answers_are_checked_against_the_facts_or_for_an_abstention() -> None:
    markers = CONFIG.evaluation.abstention_markers
    right = check_answer(QUESTION, "Servono **quarantacinque  giorni**, come dice l'Addendum n. 1.", markers)
    assert (right.facts_found, right.passed) == ((True, True), True)
    partial = check_answer(QUESTION, "Servono 45 giorni.", markers)
    assert (partial.facts_found, partial.passed) == ((True, False), False)
    assert check_answer(NO_ANSWER, "Non lo so: i documenti non riportano il numero.", markers).passed
    invented = check_answer(NO_ANSWER, "Il numero è 333 1234567.", markers)
    assert (invented.abstained, invented.passed) == (False, False)


def test_judge_verdict_is_read_from_the_reply_and_checked() -> None:
    prompts: list[str] = []

    def judge(text: str) -> ChatClient:
        def handler(request: httpx.Request) -> httpx.Response:
            prompts.append(json.loads(request.content)["messages"][-1]["content"])
            return httpx.Response(200, json={"model": "judge", "choices": [{"message": {"content": text}}]})

        return ChatClient(CONFIG.llm, "key", transport=httpx.MockTransport(handler))

    hits = [hit("EML-04_chiarimento.txt", "", "sono 45 giorni")]
    verdict = judge_faithfulness(judge('Here it is: {"claims": 4, "supported": 3}'), "answer", hits)
    assert (verdict.claims, verdict.supported, verdict.score) == (4, 3, 0.75)
    assert "[1] EML-04_chiarimento.txt | \nsono 45 giorni" in prompts[0]
    assert judge_faithfulness(judge('{"claims": 0, "supported": 0}'), "answer", hits).score == 1.0
    with pytest.raises(ValueError, match="did not reply with JSON"):
        judge_faithfulness(judge("All claims look fine."), "answer", hits)
    with pytest.raises(ValueError, match="impossible verdict"):
        judge_faithfulness(judge('{"claims": 2, "supported": 5}'), "answer", hits)
