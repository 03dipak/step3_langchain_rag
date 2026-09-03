
from pathlib import Path
from typing import Any

import pytest
from evaluator import (
    answer_relevance,
    context_precision,
    context_recall,
    evaluate_single,
    evaluate_with_registry,
    extract_keywords,
    faithful_fallback,
    faithfulness,
    relevance_fallback,
    run_full_eval,
)


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        """extract_keywords removes stopwords and short/non-alpha tokens."""
        result = extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_empty_input(self) -> None:
        """Empty string returns empty list."""
        assert extract_keywords("") == []

    def test_all_stopwords(self) -> None:
        """All-stopwords input returns empty list."""
        assert extract_keywords("the a an and or") == []

    def test_returns_lowercase(self) -> None:
        """Keywords are lowercased."""
        result = extract_keywords("Python IS Great")
        assert result == ["python", "great"]


class TestContextRecall:
    def test_all_keywords_found(self) -> None:
        """All gold keywords in context → 1.0."""
        score = context_recall(
            "chunking embeddings vector",
            ["chunking and embeddings are key to RAG", "vector stores"],
        )
        assert score == pytest.approx(1.0)

    def test_partial_keywords_found(self) -> None:
        """Partial match → fractional score."""
        score = context_recall(
            "chunking embeddings vector",
            ["chunking and embeddings are key to RAG"],
        )
        assert score == pytest.approx(2 / 3)

    def test_no_keywords_found(self) -> None:
        """No match → 0.0."""
        score = context_recall("machine learning", ["completely unrelated text"])
        assert score == 0.0

    def test_empty_gold(self) -> None:
        """Empty gold answer → 0.0."""
        assert context_recall("", ["some chunk"]) == 0.0


class TestContextPrecision:
    def test_all_relevant(self) -> None:
        """All chunks match gold_source → 1.0."""
        chunks = [
            {"text": "a", "metadata": {"source": "python_basics.txt"}},
            {"text": "b", "metadata": {"source": "python_basics.txt"}},
        ]
        assert context_precision("q", chunks, "python_basics.txt") == 1.0

    def test_half_relevant(self) -> None:
        """Half the chunks match → 0.5."""
        chunks = [
            {"text": "a", "metadata": {"source": "python_basics.txt"}},
            {"text": "b", "metadata": {"source": "other.txt"}},
        ]
        assert context_precision("q", chunks, "python_basics.txt") == 0.5

    def test_none_relevant(self) -> None:
        """No chunks match → 0.0."""
        chunks = [
            {"text": "a", "metadata": {"source": "other.txt"}},
        ]
        assert context_precision("q", chunks, "python_basics.txt") == 0.0

    def test_empty_chunks(self) -> None:
        """Empty chunks → 0.0."""
        assert context_precision("q", [], "python_basics.txt") == 0.0


class TestFaithfulFallback:
    def test_all_keywords_in_context(self) -> None:
        """All answer keywords in context → 1.0."""
        score = faithful_fallback("python is interpreted", ["python is interpreted and dynamic"])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_half_keywords_in_context(self) -> None:
        """Half answer keywords in context → 0.5."""
        score = faithful_fallback("python is compiled", ["python is interpreted and dynamic"])
        assert score == pytest.approx(0.5)

    def test_no_keywords_in_context(self) -> None:
        """No answer keywords in context → 0.0."""
        score = faithful_fallback("unknown concept", ["some unrelated text"])
        assert score == 0.0

    def test_empty_answer(self) -> None:
        """Empty answer → 0.0."""
        assert faithful_fallback("", ["some context"]) == 0.0


class TestFaithfulnessDelegatesToFallback:
    def test_faithfulness_equals_faithful_fallback(self) -> None:
        """faithfulness delegates to faithful_fallback."""
        answer = "python is interpreted"
        context = "python is interpreted and dynamic"
        assert faithfulness(answer, [context]) == faithful_fallback(answer, [context])


class TestRelevanceFallback:
    def test_identical_questions(self) -> None:
        """Identical questions → 1.0."""
        score = relevance_fallback("what is python", "what is python")
        assert score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """No keyword overlap → 0.0."""
        score = relevance_fallback("what is python", "completely different topic")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        """Partial keyword overlap → fractional score."""
        score = relevance_fallback("python programming language", "python is great")
        assert 0.0 < score < 1.0

    def test_empty_question(self) -> None:
        """Empty question → 0.0."""
        assert relevance_fallback("", "some answer") == 0.0


class TestAnswerRelevanceDelegatesToFallback:
    def test_answer_relevance_equals_relevance_fallback(self) -> None:
        """answer_relevance delegates to relevance_fallback."""
        q, a = "what is python", "python is a language"
        assert answer_relevance(q, a) == relevance_fallback(q, a)


class TestEvaluateSingle:
    def test_returns_all_metrics(self, mocker: Any) -> None:
        """evaluate_single returns a dict with all 4 metrics + answer + sources + prompt_key."""
        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "Python is interpreted.",
            "sources": [
                {"text": "Python is interpreted", "metadata": {"source": "py.txt"}},
                {"text": "dynamic typing", "metadata": {"source": "py.txt"}},
            ],
            "prompt_key": "test_key",
        }

        result = evaluate_single(
            "what is python", "Python is interpreted and dynamic", "py.txt", mock_pipeline
        )

        assert result["answer"] == "Python is interpreted."
        assert "context_recall" in result
        assert "context_precision" in result
        assert "faithfulness" in result
        assert "answer_relevance" in result
        assert result["prompt_key"] == "test_key"
        assert len(result["sources"]) == 2

    def test_pipeline_ask_called_once(self, mocker: Any) -> None:
        """evaluate_single calls pipeline.ask with the question."""
        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "test",
            "sources": [],
        }

        evaluate_single("q", "g", "src", mock_pipeline)

        mock_pipeline.ask.assert_called_once_with("q")

    def test_empty_sources_returns_zero_metrics(self, mocker: Any) -> None:
        """Empty sources still returns metrics (0.0 for recall/precision)."""
        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "test answer",
            "sources": [],
        }

        result = evaluate_single("q", "g", "src", mock_pipeline)

        assert result["context_recall"] == 0.0
        assert result["context_precision"] == 0.0
        assert result["faithfulness"] == 0.0


class TestRunFullEval:
    def test_averages_metrics(self, mocker: Any, tmp_path: Path) -> None:
        """run_full_eval averages metrics across all golden entries."""
        golden_file = tmp_path / "golden.jsonl"
        golden_file.write_text(
            '{"question": "q1", "gold_answer": "a1", "gold_source": "s1"}\n'
            '{"question": "q2", "gold_answer": "a2", "gold_source": "s2"}\n'
        )

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.side_effect = [
            {
                "answer": "ans1",
                "sources": [{"text": "chunk1", "metadata": {"source": "s1"}}],
                "prompt_key": "k1",
            },
            {
                "answer": "ans2",
                "sources": [{"text": "chunk2", "metadata": {"source": "s2"}}],
                "prompt_key": "k2",
            },
        ]

        result = run_full_eval(mock_pipeline, str(golden_file))

        assert len(result["results"]) == 2
        assert "summary" in result
        assert "context_recall" in result["summary"]
        assert "context_precision" in result["summary"]
        assert "faithfulness" in result["summary"]
        assert "answer_relevance" in result["summary"]
        assert "timestamp" in result

    def test_empty_file_returns_empty_summary(self, mocker: Any, tmp_path: Path) -> None:
        """Empty golden file → empty summary."""
        golden_file = tmp_path / "golden.jsonl"
        golden_file.write_text("")

        mock_pipeline = mocker.MagicMock()
        result = run_full_eval(mock_pipeline, str(golden_file))

        assert result["results"] == []
        assert result["summary"] == {}

    def test_preserves_individual_results(self, mocker: Any, tmp_path: Path) -> None:
        """Individual results are preserved in output."""
        golden_file = tmp_path / "golden.jsonl"
        golden_file.write_text(
            '{"question": "q1", "gold_answer": "a1", "gold_source": "s1"}\n'
        )

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "ans",
            "sources": [],
            "prompt_key": "k",
        }

        result = run_full_eval(mock_pipeline, str(golden_file))

        assert result["results"][0]["question"] == "q1"
        assert result["results"][0]["answer"] == "ans"


class TestEvaluateWithRegistry:
    def test_records_eval_scores(self, mocker: Any, tmp_path: Path) -> None:
        """evaluate_with_registry calls registry.record_eval_scores with accuracy."""
        golden_file = tmp_path / "golden.jsonl"
        golden_file.write_text(
            '{"question": "q1", "gold_answer": "a1", "gold_source": "s1"}\n'
        )

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "ans",
            "sources": [{"text": "chunk", "metadata": {"source": "s1"}}],
            "prompt_key": "pk",
        }

        mock_registry = mocker.MagicMock()
        mock_registry.REQUIRED_EVAL_KEYS = ("accuracy",)
        mock_registry.record_eval_scores = mocker.MagicMock()

        evaluate_with_registry(
            mock_pipeline, str(golden_file), mock_registry, "test_key"
        )

        mock_registry.record_eval_scores.assert_called_once()
        call_args = mock_registry.record_eval_scores.call_args
        assert call_args[0][0] == "test_key"  # prompt_key
        scores = call_args[0][1]  # scores dict
        assert "accuracy" in scores
        assert "context_recall" in scores
        assert "context_precision" in scores
        assert "faithfulness" in scores
        assert "answer_relevance" in scores

    def test_returns_results(self, mocker: Any, tmp_path: Path) -> None:
        """evaluate_with_registry returns the full results dict."""
        golden_file = tmp_path / "golden.jsonl"
        golden_file.write_text(
            '{"question": "q1", "gold_answer": "a1", "gold_source": "s1"}\n'
        )

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "ans",
            "sources": [],
            "prompt_key": "k",
        }

        mock_registry = mocker.MagicMock()
        mock_registry.REQUIRED_EVAL_KEYS = ("accuracy",)

        result = evaluate_with_registry(
            mock_pipeline, str(golden_file), mock_registry, "pk"
        )

        assert "results" in result
        assert "summary" in result
        assert "timestamp" in result
