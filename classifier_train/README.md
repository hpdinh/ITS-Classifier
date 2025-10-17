# Classifier Training

This module contains the **training pipeline** for the Service Desk classifier. It is **not deployed** in production, but is used to prepare and fine-tune models that are later served by the FastAPI app in `../app`.

---

## Directory Layout

```
classifier_train/
  data/                # Input data and processed datasets
    full_case_info.csv # Example raw dataset
  models/              # Saved model checkpoints and outputs
  src/                 # Training code
    datasets/          # Dataset definitions
      ticket_dataset.py
    utils/             # Shared utilities
      data_utils.py
      train_utils.py
    process_data.py    # Script: preprocess raw data -> train-ready format
    finetuning.py      # Script: fine-tune model
    data_config.yaml   # Config for preprocessing
    train_config.yaml  # Config for training
```

---

## Usage

1. **Preprocess data**

   ```bash
   python -m classifier_train.src.process_data \
       --config classifier_train/src/data_config.yaml \
        --input_csv classifier_train/data/full_case_info.csv
       --output_dir classifier_train/data
   ```

   This reads raw input files under `data/` and writes processed datasets for training.

2. **Train / fine-tune model**

   ```bash
   python -m classifier_train.src.finetuning \
       --config classifier_train/src/train_config.yaml
   ```

   Outputs checkpoints and logs into `models/`.

---

## Notes

* **`data/` and `models/`** are working directories. Scripts will write processed files and checkpoints here.
* **Configs (`*.yaml`)** define preprocessing and training parameters.
* This training code is intentionally separated from the serving app so deployment images remain lightweight.

---