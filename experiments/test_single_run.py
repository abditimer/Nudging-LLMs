#!/usr/bin/env python3
"""Quick test of a single experiment run"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nudging.experiment import run_single_experiment
from nudging.models import OllamaClient
from nudging.data_loader import load_data

def main():
    print("="*30)
    print("Testing single experiment run.")
    print("="*30)

    # 1. initialise model client
    print("1. initialise ollama client")
    client = OllamaClient()
    print("✓ Worked.")

    # 2. load data
    print("2. load data")
    dataset = load_data(base_dir="data", min_words=30)
    print("✓ Worked.")

    print(f"Available data:")
    print(dataset.contents.keys())
        # 'category::owner::name'
    
    # 3. test only first asset in data
    content_key = list(dataset.contents.keys())[0]
    content_text = dataset.contents[content_key]
    print(f"-> testing with {content_key}")
    print(f"-> content length: {len(content_text.split())}")

    # 4. run single experiment at 20%
    print(f"Running exp at 20% content")
    result = run_single_experiment(
        content=content_text,
        percentage=20,
        model_client=client,
        verbose=False,
    )

    # 5. display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for k,v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\n✓ Test complete!")
    print("="*60)

if __name__=="__main__":
    main()