PY ?= python

train:
	$(PY) main.py --config config.yaml

debug:
	@$(PY) main.py --config config.yaml --debug
