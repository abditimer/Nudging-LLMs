"""
My main reproducible experiment runner.

- loads data using data_loader.py
- initialises models from models.py
- runs experiments across all context percentages
- computes all metrics from metrics.py
- saves results to results/metrics as csv
- generates and saves plots to results/figures
"""

# Parse arguments
# Load config
# Load data
# Initialize model
# For each content item:
#   For each context percentage:
#     Generate continuation
#     Compute all metrics
#     Store results
# Save results DataFrame to CSV
# Generate summary plots