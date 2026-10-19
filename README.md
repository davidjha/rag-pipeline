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

## Run RAG pipeline
### Setup the document store
python setup_document_store.py

### Ingest documents
python ingest.py documents/

### Retrieve documents
python retrieve.py \
    <" Query goes here" > \
    --top-k 3 //limits the final reranked results

### Insepcting stored chunks in opensearch
curl -k \
    -u 'admin:YOUR_PASSWORD' \
    'https://localhost:9200/_cat/indices/rag-documents?v'

### View a few records at at time
curl -k \
  -u 'admin:YOUR_PASSWORD' \
  -H 'Content-Type: application/json' \
  'https://localhost:9200/rag-documents/_search?pretty' \
  -d '{
    "size": 3,
    "_source": {
      "excludes": ["embedding"]
    },
    "query": {
      "match_all": {}
    }
  }'

## Embedding Notes

The default model used is MPNet which uses 68dimensional vectors
Changing the embedding model requires:
- Change EMBEDDING_DIMENSION in config.py
- Use a new OpenSearch Index or delete and recreate the existing one.
- Re-embed all documents

Example:

OPENSEARCH_INDEX=rag-documents-<model_name>
EMBEDDING_MODEL_PATH=./models/<model_name>
EMBEDDING_DIMENSION=384

Do not mix embeddings from different models in the same vector field.

## Delete development indexZ
curl -k \
    -u 'admin:YOUR_PASSWORD' \
    -X DELETE \
    'https://localhost:9200/rag-documents'

