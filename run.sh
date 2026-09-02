#!/bin/bash
set -e

echo "=== Training ==="
python src/train.py

echo "=== Predicting ==="
python src/predict.py --data data --registry models/registry --out predictions.csv

echo "=== Baseline ==="
python baseline_3sigma.py --data data --out predictions_baseline.csv

echo "=== Validation ==="
python validate_submission.py predictions.csv
