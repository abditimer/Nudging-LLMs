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

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nudging.models import OllamaClient
from nudging.data_loader import load_data
from nudging.experiment import run_single_experiment

# load config file
from configs.experiment_config import EXPERIMENT_BASELINE
experiment_config = EXPERIMENT_BASELINE
model_config = experiment_config.model_config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


logger.info("iterating over the loaded data....")
# For each content item:
for title, content in dataset.items():
    logger.info(f"starting with: {title}")

#   For each context percentage:
    for context_percentage in experiment_config.context_percentages:
        logger.info(f"%: {context_percentage}")
        # Generate continuation
        result = run_single_experiment(
            content=content,
            percentage=context_percentage,
            model_client=client,
            verbose=False,
        )
        # Compute all metrics
        # 5. display results
        
        print("RESULTS")
        for k,v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("\n✓ Test complete!")
        print("="*60)


print("all data loaded.")

#     Store results
# Save results DataFrame to CSV
# Generate summary plots