clean-model-data:
	rm -r model_data

train:
	python3 training/train.py

run-local:
	uvicorn api:app --host 0.0.0.0 --port 80