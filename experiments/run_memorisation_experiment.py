#!/usr/bin/env python3
"""Run a configured memorisation grid and save each attempted run immediately."""

import csv
import hashlib
import json
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


RESULT_FIELDS = [
    "run_id",
    "status",
    "error",
    "text_title",
    "category",
    "model",
    "temperature",
    "seed",
    "context_percentage",
    "context_words",
    "target_words",
    "num_predict",
    "raw_generated_words",
    "generated_words",
    "raw_length_ratio",
    "scored_length_ratio",
    "exact_match",
    "fuzzy_match",
    "token_overlap",
    "semantic_similarity",
]


def _configure_file_logging(log_path: Path) -> None:
    """Add one experiment-specific log file alongside the console log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    resolved_log_path = log_path.resolve()
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in root_logger.handlers
    ):
        return

    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)


def _build_run_id(
    *,
    text_title: str,
    model: str,
    temperature: float,
    context_percentage: float,
    prompt_version: str,
    seed: int | None,
) -> str:
    """Create a stable ID for one exact experimental condition."""
    condition = {
        "text_title": text_title,
        "model": model,
        "temperature": temperature,
        "context_percentage": context_percentage,
        "prompt_version": prompt_version,
        "seed": seed,
    }
    encoded = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _completed_run_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()

    with results_path.open("r", newline="", encoding="utf-8") as results_file:
        return {
            row["run_id"]
            for row in csv.DictReader(results_file)
            if row.get("status") == "completed" and row.get("run_id")
        }


def _append_result(results_path: Path, result: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    row = {field: result.get(field) for field in RESULT_FIELDS}

    with results_path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _select_dataset(dataset: dict[str, str], selected_text_ids: Iterable[str]) -> dict[str, str]:
    selected_text_ids = list(selected_text_ids)
    if not selected_text_ids:
        raise ValueError("selected_text_ids must contain at least one text ID.")

    missing_ids = [text_id for text_id in selected_text_ids if text_id not in dataset]
    if missing_ids:
        raise KeyError(f"Configured text IDs were not found: {missing_ids}")

    return {text_id: dataset[text_id] for text_id in selected_text_ids}


def _category_from_title(text_title: str) -> str:
    return text_title.split("::", maxsplit=1)[0]


def run_experiment(
    experiment_config,
    dataset: dict[str, str],
    results_path: Path,
    max_runs: int | None = None,
) -> None:
    """Run all configured conditions, appending each completed or failed row."""
    from nudging.experiment import run_experiments
    from nudging.models import OllamaClient

    selected_dataset = _select_dataset(dataset, experiment_config.selected_text_ids)
    completed_ids = _completed_run_ids(results_path)
    total_runs = (
        len(selected_dataset)
        * len(experiment_config.models)
        * len(experiment_config.temperatures)
        * len(experiment_config.context_percentages)
    )
    attempted = 0
    skipped = 0
    completed = 0
    errors = 0

    logger.info(
        "Starting %s: %s planned runs (%s texts × %s models × %s temperatures × %s contexts)",
        experiment_config.name,
        total_runs,
        len(selected_dataset),
        len(experiment_config.models),
        len(experiment_config.temperatures),
        len(experiment_config.context_percentages),
    )
    logger.info("Results CSV: %s", results_path)
    if max_runs is not None:
        logger.info("Run limit: %s newly attempted condition(s)", max_runs)
    logger.info("Seed=%s | token_multiplier=%s | semantic=%s",
                experiment_config.random_seed,
                experiment_config.token_multiplier,
                experiment_config.include_semantic)
    logger.info("Selected text IDs: %s", list(selected_dataset))
    for model_config in experiment_config.models:
        logger.info("Initialising model: %s (%s)", model_config.name, model_config.endpoint)
        client = OllamaClient(
            model=model_config.name,
            base_url=model_config.endpoint,
        )
        if not client.ensure_running():
            raise RuntimeError(f"Ollama is unavailable for model {model_config.name!r}.")
        logger.info("Ollama is ready for %s", model_config.name)

        for temperature in experiment_config.temperatures:
            for text_title, content in selected_dataset.items():
                for context_percentage in experiment_config.context_percentages:
                    if max_runs is not None and attempted >= max_runs:
                        logger.info("Reached run limit; stopping before the next condition.")
                        logger.info(
                            "Finished %s: completed=%s errors=%s skipped=%s results=%s",
                            experiment_config.name,
                            completed,
                            errors,
                            skipped,
                            results_path,
                        )
                        return

                    run_id = _build_run_id(
                        text_title=text_title,
                        model=model_config.name,
                        temperature=temperature,
                        context_percentage=context_percentage,
                        prompt_version=experiment_config.prompt_version,
                        seed=experiment_config.random_seed,
                    )
                    if run_id in completed_ids:
                        skipped += 1
                        logger.info("Skipping completed run %s", run_id)
                        continue

                    attempted += 1
                    logger.info(
                        "Starting run %s: model=%s temperature=%s text=%s context=%s%%",
                        run_id,
                        model_config.name,
                        temperature,
                        text_title,
                        context_percentage,
                    )
                    base_result = {
                        "run_id": run_id,
                        "text_title": text_title,
                        "category": _category_from_title(text_title),
                        "model": model_config.name,
                        "temperature": temperature,
                        "seed": experiment_config.random_seed,
                        "context_percentage": context_percentage,
                    }
                    try:
                        metrics = run_experiments(
                            title=text_title,
                            content=content,
                            percentage=context_percentage,
                            model_client=client,
                            prompt_version=experiment_config.prompt_version,
                            temperature=temperature,
                            seed=experiment_config.random_seed,
                            token_multiplier=experiment_config.token_multiplier,
                            include_semantic=experiment_config.include_semantic,
                        )
                        result = {**base_result, **metrics, "status": "completed", "error": ""}
                        completed_ids.add(run_id)
                        completed += 1
                    except Exception as exc:
                        logger.exception("Run %s failed", run_id)
                        result = {
                            **base_result,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        errors += 1

                    _append_result(results_path, result)
                    logger.info(
                        "Saved %s row %s/%s (%s skipped): raw_words=%s generated_words=%s",
                        result["status"],
                        attempted,
                        total_runs,
                        skipped,
                        result.get("raw_generated_words"),
                        result.get("generated_words"),
                    )

                    if experiment_config.context_delay_seconds > 0:
                        logger.info("Waiting %.1f seconds before the next run", experiment_config.context_delay_seconds)
                        time.sleep(experiment_config.context_delay_seconds)

    logger.info(
        "Finished %s: completed=%s errors=%s skipped=%s results=%s",
        experiment_config.name,
        completed,
        errors,
        skipped,
        results_path,
    )


def _parse_args(config_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a named memorisation experiment configuration.",
    )
    parser.add_argument(
        "--config",
        choices=config_names,
        default="smoke",
        help="Named configuration to run (default: smoke).",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configuration names and exit.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after this many newly attempted conditions; useful for smoke checks.",
    )
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be a positive integer.")
    return args


def _setup_experiment_for_terminal(config_name: str):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from configs.experiment_config import EXPERIMENT_CONFIGS
    from nudging.data_loader import load_data

    experiment_config = EXPERIMENT_CONFIGS[config_name]
    dataset = load_data(
        base_dir=project_root / experiment_config.data_config.data_folder_name,
        min_words=experiment_config.data_config.min_word_count,
        categories=experiment_config.data_config.categories,
    )
    results_path = project_root / "results" / "metrics" / experiment_config.output_filename
    log_path = project_root / "results" / "logs" / f"{experiment_config.name}.log"
    return experiment_config, dataset, results_path, log_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from configs.experiment_config import EXPERIMENT_CONFIGS

    args = _parse_args(sorted(EXPERIMENT_CONFIGS))
    if args.list_configs:
        for config_name, config in EXPERIMENT_CONFIGS.items():
            print(f"{config_name}: {config.name}")
        raise SystemExit(0)

    experiment_config, dataset, results_path, log_path = _setup_experiment_for_terminal(args.config)
    _configure_file_logging(log_path)
    logger.info("Selected terminal configuration: %s", args.config)
    logger.info("Writing execution log to %s", log_path)
    run_experiment(
        experiment_config,
        dataset,
        results_path,
        max_runs=args.max_runs,
    )
