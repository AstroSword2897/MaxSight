#!/bin/bash
# Helper script to ensure scripts run with the correct venv Python
# Usage: ./scripts/run_with_venv.sh <script> [args...]

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to venv (as configured in pyproject.toml)
VENV_PATH="${REPO_ROOT}/../2026/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it with: python3.10 -m venv $VENV_PATH"
    exit 1
fi

# Activate venv and run the script
source "$VENV_PATH/bin/activate"

# Verify we're using the venv Python
PYTHON_EXEC=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_EXEC" != "$VENV_PATH/bin/python"* ]]; then
    echo "Warning: Not using venv Python. Current: $PYTHON_EXEC"
    echo "Expected: $VENV_PATH/bin/python"
fi

# Run the provided script with all arguments
exec python "$@"

