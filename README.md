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
- **Ruff**: Fast Python linter and code formatter for maintaining code quality and consistency


## Data

The Watkins Marine Mammal Sound Database (WMMS) is a comprehensive collection of marine mammal vocalizations curated by the Woods Hole Oceanographic Institution. This dataset contains over 1,600 cuts (audio segments) spanning 32,000 hours of recordings from 60+ marine species including whales, dolphins, seals, and other marine mammals. Each audio sample is labeled with species information and represents diverse acoustic characteristics of marine mammal communication. The dataset has been converted to Parquet format for efficient processing and is split into training and test sets.

## Expected Models

The expected model we're going to use is the ResNet34, a variant of the ResNet family which consists of deep convolutional neural networks originally developed for image recognition tasks. It has 34 layers balancing complexity and computational efficiency. In the context of audio classification, we can transform raw audio recordings into image representations, such as spectrograms, which encode the frequency and temporal information of the sounds. By feeding these spectrograms into ResNet34, the model can automatically learn patterns and features corresponding to different animal vocalizations.

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

# Docker Usage

## Building the images

All build commands must be executed from the project root directory.

| Image Type | Dockerfile |
| :--- | :--- |
| **Training (CPU)** | `dockerfiles/train_cpu.dockerfile` |
| **Training (GPU)** | `dockerfiles/train_gpu.dockerfile` |
| **Evaluation** | `dockerfiles/evaluate.dockerfile` |

### Build commands

**1. CPU training image**
For local testing.
```bash
docker build -f dockerfiles/train_cpu.dockerfile . -t proj_train_cpu:latest
```

**2. GPU training image**
Uses NVIDIA's PyTorch base image. Requires NVIDIA drivers.
```bash
docker build -f dockerfiles/train_gpu.dockerfile . -t proj_train_gpu:latest
```

**3. Evaluation image**
For running model benchmarks.
```bash
docker build -f dockerfiles/evaluate.dockerfile . -t proj_eval:latest
```

---

## Running the containers

### 1. CPU training

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/models:/models \
  -v $(pwd)/reports:/reports \
  proj_train_cpu:latest
```

### 2. GPU Accelerated Training

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/models:/models \
  -v $(pwd)/reports:/reports \
  proj_train_gpu:latest
```

### 3. Model evaluation

```bash
docker run --rm \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  proj_eval:latest
```



Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
