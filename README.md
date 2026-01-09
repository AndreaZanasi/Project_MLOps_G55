# Classifying animal vocalizations

## Overall goal
The objective of this project is to classify animal vocalizations. By utilizing audio recordings of various species, such as birds, dogs, bats, and orcas, we aim to transform raw audio into spectrograms and treat the task as an image classification problem.

## Framework

- **PyTorch**: Deep learning framework for building and training neural networks
- **Hugging Face Datasets**: Library for easy access and processing of the animal sounds dataset
- **torchaudio**: Audio processing library for converting audio files to spectrograms
- **PyTorch Lightning**: Framework for reducing boilerplate code and structuring training workflows
- **Weights & Biases (wandb)**: Experiment tracking and logging platform
- **uv**: Fast Python package manager for dependency management
- **Hydra**: Configuration management framework for composing and overriding configs dynamically
- **Torch Profiler**: Performance analysis tool for profiling PyTorch models and identifying bottlenecks
- **Ruff**: Fast Python linter and code formatter for maintaining code quality and consistency


## Data

The Animal Sounds Collection (cgeorgiaw/animal‑sounds) is a multi-species dataset designed for species classification. It contains annotated recordings of vocalizations from 7 different animals (birds, dogs, egyptian fruit baits, giant otters, macaques, orcas and zebra finches). The dataset is organized modularly with separate splits per species, each recording is labeled with the corresponding species,moreover the recordings vary in length and acoustic characteristics.

## Expected Models

The expected model we're going to use is the ResNet38, a variant of the ResNet family which consists of deep convolutional neural networks originally developed for image recognition tasks. It has 38 layers balancing complexity and computational efficiency. In the context of audio classification, we can transform raw audio recordings into image representations, such as spectrograms, which encode the frequency and temporal information of the sounds. By feeding these spectrograms into ResNet38, the model can automatically learn patterns and features corresponding to different animal vocalizations.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
