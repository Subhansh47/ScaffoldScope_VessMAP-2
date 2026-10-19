#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Install Python 3.11 or 3.12 from https://www.python.org/downloads/macos/"
  exit 1
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Next: prepare paired images and masks in data/vessmap/images and data/vessmap/masks."
