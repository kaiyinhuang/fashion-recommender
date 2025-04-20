#!/bin/bash
uvicorn api.app:app --host $(jq -r .api.host configs/paths.yaml) --port $(jq -r .api.port configs/paths.yaml)
