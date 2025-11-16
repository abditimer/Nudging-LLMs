#!/bin/bash
# Simple test runner script

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the tests
python3 -m unittest tests.test_models -v
