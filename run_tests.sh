#!/bin/bash
# Simple test runner script

# Prefer the repository's dependency-managed environment when it exists.
if [ -x ".venv.nosync/bin/python" ]; then
    PYTHON_BIN=".venv.nosync/bin/python"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
"${PYTHON_BIN}" -m unittest discover -s tests -v
