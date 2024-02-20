clean:
	rm -r model_data

train:
	python3 classifier_training.py

local:
	uvicorn classifier_api:app --host 0.0.0.0 --port 80

run_all:
	mkdir model_data
	python3 classifier_training.py
	uvicorn classifier_api:app --host 0.0.0.0 --port 80
