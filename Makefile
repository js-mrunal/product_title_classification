clean-env:
	pipenv --rm

format:
	pipenv run isort .
	pipenv run black .

init:
	pipenv install --dev

clean-model-data:
	rm -r model_data

train:
	pipenv run python training/train.py

run-local:
	pipenv run uvicorn api:app --host 0.0.0.0 --port 8080