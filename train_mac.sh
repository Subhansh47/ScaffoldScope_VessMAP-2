#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Environment missing. Run ./setup_mac.sh first."
  exit 1
fi
source .venv/bin/activate
python train.py --epochs "${1:-2}" --batch-size 2 --size 256 --device cpu
