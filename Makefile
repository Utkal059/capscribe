.PHONY: install test lint serve eval index

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn api:app --reload --port 8000

eval:
	python evaluate.py fixtures/sample_events.json fixtures/gold_events.json

# Rebuild the persistent index from a chosen extracted file
index:
	python -c "from retrieval import EventStore, load_events; \
		n=EventStore().index_events(load_events('$(FILE)')); print('indexed', n)"
