# Python interpreter
PYTHON := python3

# Virtual environment directory
VENV_DIR := .venv

# Requirements file
REQUIREMENTS := requirements.txt

# Target to create a virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "To activate it, run 'source $(VENV_DIR)/bin/activate'"

# Target to install dependencies
install: venv 
	source $(VENV_DIR)/bin/activate && echo "Virtual environment activated"
	@echo "Installing dependencies..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)
	@echo "Dependencies installed."

# Target to export current dependencies to requirements.txt
freeze:
	@echo "Exporting dependencies to $(REQUIREMENTS)..."
	pip freeze > $(REQUIREMENTS)
	@echo "Dependencies exported to $(REQUIREMENTS)."

# Target to clean the virtual environment
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleaned up."

# Start main.py
start:
	$(PYTHON) main.py

# Target to show available commands
help:
	@echo "Available commands:"
	@echo "  make venv       - Create a virtual environment"
	@echo "  make install    - Install dependencies from requirements.txt"
	@echo "  make freeze     - Export installed dependencies to requirements.txt"
	@echo "  make clean      - Remove the virtual environment"
	@echo "  make start      - Start main.py"
