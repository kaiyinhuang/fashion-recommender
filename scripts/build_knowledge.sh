#!/bin/bash

echo "Building fashion knowledge base..."
python -m core.knowledge_builder \
    --input data/knowledge/raw_dialogs.csv \
    --output knowledge/ \
    --config configs/model_config.yaml

echo "Validating knowledge base..."
pytest tests/test_rag.py
