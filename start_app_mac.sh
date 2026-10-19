#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Environment missing. Run ./setup_mac.sh first."
  exit 1
fi
source .venv/bin/activate
python -m streamlit run app.py --server.port 8765
