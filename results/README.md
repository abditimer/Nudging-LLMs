# Results Directory

This directory contains all experimental outputs from the memorisation study.

## Structure

### `figures/`
Publication-ready plots and visualizations

### `metrics/`
CSV files with raw experimental data

### `logs/`
Experiment execution logs

## Reproducing Results

To regenerate all results:
```bash
python experiments/run_memorization_experiment.py --config configs/base_config.yaml
python experiments/evaluate_results.py --input results/metrics/
