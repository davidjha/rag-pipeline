# rag-pipeline


## Setup python environment

### Create a virtual environment

python venv .rag-pipeline

source .rage-pipeline/bin/activate

### Install python packages

python pip install -r requirments.txt

## Save python package for airgapped systems

python -m pip freeze > requirements.txt
python -m pip download \
    --only-binary=:all: \
    --requirement requirements.txt \
    --dest wheelhouse

## Install python packages for airgrapped systems

python -m pip install \
    --no-index \
    --find-links=wheelhouse \
    --requirement requirements.txt

If packages have no published wheel, build wheels on the source machine.

python -m pip wheel \
    --requirement requirements.txt
    --wheel-dir wheelhouse

Note:

Compiled wheels are typically specific to:
- Operating System
- CPU architecture
- Python implementation and version
- Systems libraries
