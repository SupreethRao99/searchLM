.PHONY: help clean clean-pyc clean-build clean-test clean-cache organize-imports check-imports format lint check all

# Default target
help:
	@echo "Available commands:"
	@echo "  make clean              - Remove all build, cache, and test artifacts"
	@echo "  make clean-pyc          - Remove Python cache files (__pycache__, *.pyc)"
	@echo "  make clean-build        - Remove build artifacts (build/, dist/, *.egg-info)"
	@echo "  make clean-test         - Remove test artifacts (.pytest_cache, .coverage, htmlcov)"
	@echo "  make clean-cache        - Remove cache directories (.ruff_cache, .mypy_cache)"
	@echo "  make organize-imports   - Organize and sort imports in all Python files"
	@echo "  make check-imports      - Check if imports are organized (dry-run)"
	@echo "  make format             - Format code and organize imports using ruff"
	@echo "  make lint               - Lint code using ruff"
	@echo "  make check              - Check imports and lint without modifying files"
	@echo "  make all                - Clean, organize imports, format, and lint"

# Clean all artifacts
clean: clean-pyc clean-build clean-test clean-cache
	@echo "✓ All cleaned!"

# Remove Python cache files
clean-pyc:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	@find . -type f -name "*$$py.class" -delete 2>/dev/null || true
	@find . -type f -name "*.so" -delete 2>/dev/null || true
	@echo "✓ Python cache files cleaned"

# Remove build artifacts
clean-build:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info .eggs/ develop-eggs/ downloads/ eggs/ lib/ lib64/ parts/ sdist/ var/ wheels/ share/python-wheels/ 2>/dev/null || true
	@rm -f MANIFEST *.egg 2>/dev/null || true
	@echo "✓ Build artifacts cleaned"

# Remove test artifacts
clean-test:
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache/ .coverage htmlcov/ .tox/ .cache/ 2>/dev/null || true
	@echo "✓ Test artifacts cleaned"

# Remove cache directories
clean-cache:
	@echo "Cleaning cache directories..."
	@rm -rf .ruff_cache/ .mypy_cache/ .ipynb_checkpoints/ 2>/dev/null || true
	@echo "✓ Cache directories cleaned"

# Organize imports in all Python files
organize-imports:
	@echo "Organizing imports in all Python files..."
	@uv run isort searchlm/ scripts/ --check-only --diff || uv run isort searchlm/ scripts/
	@echo "✓ Imports organized"

# Check if imports are organized (dry-run)
check-imports:
	@echo "Checking import organization..."
	@uv run isort searchlm/ scripts/ --check-only --diff
	@echo "✓ Import check complete"

# Format code and organize imports using ruff
format:
	@echo "Formatting code and organizing imports..."
	@uv run ruff format searchlm/ scripts/
	@uv run ruff check --fix --unsafe-fixes searchlm/ scripts/ || true
	@uv run isort searchlm/ scripts/
	@echo "✓ Code formatted"

# Lint code using ruff
lint:
	@echo "Linting code..."
	@uv run ruff check searchlm/ scripts/
	@echo "✓ Linting complete"

# Check imports and lint without modifying files
check: check-imports lint
	@echo "✓ All checks complete"

# Clean, organize imports, format, and lint
all: clean organize-imports format lint
	@echo "✓ All tasks complete!"
