import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run_memorisation_experiment import (
    RESULT_FIELDS,
    _append_result,
    _build_run_id,
    run_experiment,
)
from nudging.experiment import _generate_response, _get_num_predict_for_target, _trim_to_n_words, run_experiments
from nudging.prompt import build_continuation_prompt


class FakeModelClient:
    def __init__(self, response="five six seven eight nine"):
        self.response = response
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append({"prompt": prompt, "options": options})
        return self.response


class TestExperimentLengthControl(unittest.TestCase):
    def test_v4_prompt_contains_context_and_target_length(self):
        prompt = build_continuation_prompt("v4", "one two", 7)
        self.assertIn("one two", prompt)
        self.assertIn("approximately 7 whitespace-separated words", prompt)
        self.assertIn("Never exceed 7", prompt)

    def test_unknown_prompt_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown prompt version"):
            build_continuation_prompt("unknown", "context", 3)

    def test_trim_to_n_words(self):
        self.assertEqual(_trim_to_n_words("one two three four", 2), "one two")

    def test_num_predict_uses_token_multiplier(self):
        self.assertEqual(_get_num_predict_for_target(10, 1.5), 15)
        self.assertEqual(_get_num_predict_for_target(11, 1.5), 17)

    def test_generation_forwards_options_and_trims(self):
        client = FakeModelClient()
        generated, context, target, metadata = _generate_response(
            content="one two three four",
            percentage=50,
            model_client=client,
            prompt_version="v4",
            temperature=0.0,
            seed=42,
            token_multiplier=1.5,
        )

        self.assertEqual((context, target, generated), ("one two", "three four", "five six"))
        self.assertEqual(client.calls[0]["options"], {"temperature": 0.0, "seed": 42, "num_predict": 3})
        self.assertEqual(metadata["raw_generated_words"], 5)
        self.assertEqual(metadata["generated_words"], 2)

    def test_run_returns_configured_scores(self):
        result = run_experiments(
            title="songs::artist::title",
            content="one two three four",
            percentage=50,
            model_client=FakeModelClient("three four five"),
            prompt_version="v4",
            temperature=0.0,
            seed=42,
            token_multiplier=1.5,
            include_semantic=False,
        )
        self.assertTrue({"exact_match", "fuzzy_match", "token_overlap", "semantic_similarity"} <= result.keys())
        self.assertIsNone(result["semantic_similarity"])


class TestBatchRunner(unittest.TestCase):
    def test_run_id_is_stable_and_changes_with_condition(self):
        args = dict(
            text_title="songs::artist::title", model="qwen2.5:0.5b-instruct",
            temperature=0.0, context_percentage=50, prompt_version="v4", seed=42,
        )
        self.assertEqual(_build_run_id(**args), _build_run_id(**args))
        self.assertNotEqual(_build_run_id(**args), _build_run_id(**{**args, "temperature": 0.7}))

    def test_append_result_uses_only_configured_csv_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"
            _append_result(path, {"run_id": "id", "status": "completed", "extra": "not saved"})
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(tuple(rows[0]), tuple(RESULT_FIELDS))
        self.assertEqual(rows[0]["run_id"], "id")

    def test_resume_skips_completed_run(self):
        config = SimpleNamespace(
            name="test", models=[SimpleNamespace(name="model", endpoint="http://unused")],
            temperatures=[0.0], context_percentages=[50], random_seed=42,
            prompt_version="v4", token_multiplier=1.5, include_semantic=False,
            selected_text_ids=["songs::artist::title"], context_delay_seconds=0.0,
        )
        run_id = _build_run_id(
            text_title="songs::artist::title", model="model", temperature=0.0,
            context_percentage=50, prompt_version="v4", seed=42,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "results.csv"
            _append_result(results_path, {"run_id": run_id, "status": "completed"})
            with patch("nudging.models.OllamaClient") as client_class, patch(
                "nudging.experiment.run_experiments",
                side_effect=AssertionError("a completed run must not generate"),
            ):
                client_class.return_value.ensure_running.return_value = True
                run_experiment(config, {"songs::artist::title": "one two three four"}, results_path)

        self.assertFalse(client_class.return_value.generate.called)


if __name__ == "__main__":
    unittest.main()
