#!/usr/bin/env python3
"""
My main reproducible experiment runner.

- loads data using data_loader.py
- initialises models from models.py
- runs experiments across all context percentages
- computes all metrics from metrics.py
- saves results to results/metrics as csv
- generates and saves plots to results/figures
"""
import json
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(experiment_config, model_config, client, dataset):
    from nudging.experiment import run_experiments
    
    experiment_results = []

    logger.info("iterating over the loaded data....")
    # For each content item:
    for title, content in dataset.items():
        logger.info(f"starting with: {title}")

    #   For each context percentage:
        for context_percentage in experiment_config.context_percentages:
            logger.info(f"=====>{context_percentage}%")
            # Generate continuation
            result = run_experiments(
                title=title,
                content=content,
                percentage=context_percentage,
                model_client=client,
                verbose=False,
            )

            # TODO: look at metrics logic being returned
            logger.info("Experiment results: %s", json.dumps(result, indent=2))

            experiment_results.append(result)

    return experiment_results

def _setup_experiment_for_terminal():
    import sys
    from pathlib import Path
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from nudging.models import OllamaClient
    from nudging.data_loader import load_data
   

    # load config file
    from configs.experiment_config import EXPERIMENT_BASELINE
    experiment_config = EXPERIMENT_BASELINE
    model_config = experiment_config.model_config

    logger.info(f"Running: {experiment_config.name}")
    logger.info(f"Context percentages: {experiment_config.context_percentages}")

    # Initialise model
    logger.info("initialise ollama client...")
    client = OllamaClient(model=model_config.name)
    logger.info("✓")

    # initialise data
    logger.info("loading data...")
    dataset = load_data(
        base_dir=experiment_config.data_config.data_folder_name, 
        min_words=experiment_config.data_config.min_word_count, 
        max_samples=experiment_config.max_samples
    )
    logger.info("✓")

    return experiment_config, model_config, client, dataset

if __name__ == "__main__":
    experiment_config, model_config, client, dataset = _setup_experiment_for_terminal()
    run_experiment(experiment_config, model_config, client, dataset)