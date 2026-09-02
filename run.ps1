Write-Host "=== Training ==="
python src/train.py --registry models/registry

Write-Host "=== Predicting ==="
python src/predict.py --data data --registry models/registry --out predictions.csv

Write-Host "=== Baseline ==="
python baseline_3sigma.py --data data --out predictions_baseline.csv

Write-Host "=== Validation ==="
python validate_submission.py predictions.csv
