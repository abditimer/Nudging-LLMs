# Nudging: How Memorization of Large Language Models Scale

Research Question
How do LLMs handle creative content released after their training cutoff date?
We investigate completion accuracy across content types (e.g. songs, podcasts, books), model providers (OpenAI, IBM, open-source) and varying prefix length.

This is the first systematic,cross-modal, cross-model emperical study of LLM behaviour on guaranteed post-training content. This provides practical guidance on LLM capabilities for new creative works and sets a baseline for contamination detection.

## Overview

Primary Research Questions:
1. How accurate can LLMs complete creative content released after their training cutoff?
hypothesis: Completion accuracy will be near-zero for exact matches (<5%)
2. Does completion accuracy vary by content type?
hypothesis: Songs will show higher completion than prose
3. Does completion accuracy vary by model provider?
hypothesis: commercial models (GPT5, Claude) will outperform closed models
4. How does prefix length affect completion accuract for unseen content?
hypothesis: Longer prefix length provides more context but does not enable memorisation

### Overview of this project

You'll find all my code in the following
- nudging/ : this is our main source code package
- experiments/ : experiment runner scripts
- configs/ : configuration files

You'll find research artifacts here
- data/ : **not included.**
- notebooks/ exploratoration
- results/ : experiment results, figures, tables
- models/ : trained model weights

for reproducibility:
- tests/ : unit tests for core functionality


## Dataset Setup

The dataset is not included in this repository. To run experiments:

1. Create the data directory structure:
```bash
mkdir -p data/podcasts data/songs
Add your text files following this structure:
data/
├── podcasts/
│   └── {owner}/
│       └── {name}.txt
└── songs/
    └── {owner}/
        └── {name}.txt

## Installation

### Requirements
- Python 3.14
- Ollama (for local model inference)

### Setup
```bash
# Clone and install
pip install -r requirements.txt

# Or use Docker
docker-compose up

Usage
Running Experiments
python experiments/run_memorization_experiment.py --model qwen3:0.6b
Reproducing Results
[Instructions to reproduce paper results]
Project Structure
[Brief overview of directories]
Dataset
[Description of data categories - note: actual data not included for privacy]
Models Tested
qwen3:0.6b
[List other models]
Results
[Link to results/ directory or summary of key findings]
Citation
[Your citation]
License
[MIT/Apache 2.0]