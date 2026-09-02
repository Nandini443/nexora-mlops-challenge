# Run the full pipeline end-to-end

DATA_DIR = data
REGISTRY_DIR = models/registry
PREDICTIONS = predictions.csv
BASELINE = predictions_baseline.csv

train:
    python src/train.py --registry $(REGISTRY_DIR)

predict:
    python src/predict.py --data $(DATA_DIR) --registry $(REGISTRY_DIR) --out $(PREDICTIONS)

baseline:
    python baseline_3sigma.py --data $(DATA_DIR) --out $(BASELINE)

validate:
    python validate_submission.py $(PREDICTIONS)

all: train predict baseline validate
