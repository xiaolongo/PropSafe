**PropSafe** is a Python-based graph out-of-distribution framework. It provides a structured codebase for implementing, training, and assessing models.

## Features

- **Modular Architecture**: Clean separation between backbone networks (`backbone.py`), core models (`model.py`), and training/testing logic for easy modification and extension.
- **Standardized Workflow**:
  - `train_id.py`: Handles standard training on In-Distribution (ID) data.
  - `test_ood.py`: Dedicated script for evaluating model generalization under Out-of-Distribution (OOD) scenarios.
- **Flexible Configuration**: Centralized hyperparameter and experiment management via `args.py`.
- **Experiment Logging**: Integrated utilities in `logger.py` for systematic tracking of metrics and results.

## Repository Structure

```
PropSafe/
├── args.py          # Argument parsing and configuration definitions
├── backbone.py      # Feature extractor backbone definitions (e.g., ResNet, ViT)
├── logger.py        # Logging and experiment tracking utilities
├── model.py         # Core model architecture definition
├── train_id.py      # Standard In-Distribution training script
├── test_ood.py      # Out-of-Distribution evaluation script
├── LICENSE          # License file
└── README.md        # Project documentation
```

### Train the Model
Run a standard training session with default parameters:
```bash
python train_id.py
```

### Evaluate the Model
After training, assess the model's OOD performance using the test script:
```bash
python test_ood.py
```

## Configuration
Key parameters are defined in `args.py`. Commonly adjusted options include:
- `--dataset`: Specify the training dataset.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training and evaluation.
- `--lr`: Learning rate.
- `--backbone`: Type of feature extraction network.
