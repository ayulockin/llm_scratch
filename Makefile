# Define your Python files or directories to check
PY_FILES := src/

# Default rule: run all checks
all: black mypy isort ruff

# Run black for code formatting
black:
	@echo "Running Black..."
	black $(PY_FILES)

# Run mypy for type checking
mypy:
	@echo "Running MyPy..."
	mypy $(PY_FILES)

# Run isort for import sorting
isort:
	@echo "Running isort..."
	isort $(PY_FILES)

# Run ruff for linting
ruff:
	@echo "Running ruff..."
	ruff $(PY_FILES)

# Help message
help:
	@echo "Makefile commands:"
	@echo "  make          Run all checks (black, mypy, isort, ruff)"
	@echo "  make black    Run Black for code formatting"
	@echo "  make mypy     Run MyPy for type checking"
	@echo "  make isort    Run isort for import sorting"
	@echo "  make ruff     Run Ruff for linting"
