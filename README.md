# MLLM Emotion and Affect Recognition

## Environment Setup

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management. Make sure `uv` is installed before proceeding.

ffmpeg must also be loaded as a system module:

```bash
module load ffmpeg
```

### Install dependencies

A `uv.lock` file is included in the repository. To create a virtual environment and install all pinned dependencies:

```bash
uv sync
```

This will create a `.venv` directory in the project root and install all packages from the lock file.

### Activate the environment

```bash
source .venv/bin/activate
```

Or run commands directly without activating:

```bash
uv run python main.py
```

### Full setup sequence

```bash
module load ffmpeg
uv sync
source .venv/bin/activate
```
