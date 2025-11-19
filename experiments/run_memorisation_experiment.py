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
from configs.experiment_config import MEMORISATION_BASELINE, DataConfig, ModelConfig
experiment = MEMORISATION_BASELINE
data_config = DataConfig()
model_config = ModelConfig()

print(f"Running: {experiment.name}")
print(f"Context percentages: {experiment.context_percentages}")

# Initialise model
print("initialise ollama client...")
client = OllamaClient()
print("OK")

# initialise data
print("loading data...")
dataset = load_data(base_dir="data", min_words=30)
print(dataset.summary)
print("OK")

# For each content item:
for title, content in dataset.contents.items():
    print(f"{title}: {content[:10]}")
#   For each context percentage:
    for context_percentage in experiment.context_percentages:
        print("\n" + "="*60)
        print(context_percentage)
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

#     Store results
# Save results DataFrame to CSV
# Generate summary plots