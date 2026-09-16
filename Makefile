.PHONY: help install test pipeline retrain reproduce run-api

help:
	@echo "======================================================================"
	@echo "  @AmazonHelp AI Support Agent - Developer & Reproduction Commands"
	@echo "======================================================================"
	@echo "  make install    : Install bounded dependencies from requirements.txt"
	@echo "  make test       : Execute full automated pytest test suite"
	@echo "  make pipeline   : Run master evaluation pipeline against golden set"
	@echo "  make retrain    : Force rebuild of indexes and retrain ML model"
	@echo "  make reproduce  : 1-Click full verification (env + tests + pipeline)"
	@echo "  make run-api    : Launch FastAPI microservice on http://0.0.0.0:8000"
	@echo "======================================================================"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

pipeline:
	python run_pipeline.py

retrain:
	python run_pipeline.py --force-retrain

reproduce:
	python reproduce.py

run-api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
