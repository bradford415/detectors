## Makefile to automate commands like ci/cd, installing dependencies, and creating a virtual environment

## Excecutes all commands in a single shell (allows us to activate a virtual env and install dependencies in it)
.ONESHELL:

.PHONY: venv test require install create

## Variables set at declaration time
PROJECT_NAME := detectors
VENV_NAME := detectors
REQUIREMENTS := requirements.txt

## Recursively expanded variables
python_source = ${PROJECT_NAME} scripts/  # Location of python files 
activate = source .venv/bin/activate
activate_windows = source .venv/Scripts/activate

python = python3

venv: ## Create virtual environment
	${python} -m venv .venv

test: ## Put pytests here
	. 
	pytest tests/

format: ## Reformats your code to a specific coding style
	${activate}
	black ${python_source}
	isort ${python_source}

format_windows:
	${activate_windows}
	black ${python_source}
	isort ${python_source} 

check: ## Check to see if any linux formatting 
	${activate}
	black --check ${python_source}
	isort --check-only ${python_source}
	mypy ${python_source}
	pylint ${python_source}

check_windows:
	${activate_windows}
	black --check ${python_source}
	isort --check-only ${python_source}
	mypy ${python_source}
	pylint ${python_source}

require:
	pip install pip-tools
	pip-compile --output-file requirements.txt pyproject.toml 

install_deps: ## Install for linux only; we also need to upgrade pip to support editable installation with only pyproject.toml file
	${activate}
	python -m pip install --upgrade pip
	python -m pip install -r ${REQUIREMENTS}
 	python -m pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	python -m ${python} -m pip install -e . --no-deps

install_deps_windows:
	${activate_windows}
	python -m pip install --upgrade pip
	python -m pip install -r ${REQUIREMENTS}
	python -m ${python} -m pip install -e . --no-deps

create: venv install_deps ## Create virtual environment and install dependencies and the project itself

create_windows: venv install_deps_windows
