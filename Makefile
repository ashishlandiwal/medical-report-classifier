.PHONY: install dev test lint train app docker-build docker-run

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

train:
	PYTHONPATH=src python -m medclf.train --no-mlflow --output reports

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t medical-report-classifier .

docker-run:
	docker run --rm -p 8501:8501 medical-report-classifier
