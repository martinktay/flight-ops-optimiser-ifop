# Flight Operations Optimiser Makefile
# Common development tasks and utilities

.PHONY: help install install-dev test test-cov lint format type-check clean build docker-build docker-run docs

# Default target
help:
	@echo "Flight Operations Optimiser - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run unit tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  test-int     - Run integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  type-check   - Run type checking with mypy"
	@echo ""
	@echo "Development:"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "Pipeline:"
	@echo "  pipeline     - Run full pipeline"
	@echo "  dagster-dev  - Start Dagster development server"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov pytest-mock black flake8 mypy pre-commit

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-int:
	pytest tests/ -m integration -v

# Code Quality
lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black src/ tests/ --line-length=88

format-check:
	black src/ tests/ --check --diff --line-length=88

type-check:
	mypy src/ --ignore-missing-imports

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docs:
	cd docs && make html

# Docker
docker-build:
	docker build -t flight-ops-optimiser:latest .

docker-run:
	docker run -p 8000:8000 flight-ops-optimiser:latest

docker-run-dev:
	docker run -p 8000:8000 -p 3000:3000 --target development flight-ops-optimiser:latest

# Pipeline
pipeline:
	python -m src.main --mode full --start-date 2024-01-01 --end-date 2024-01-31 --airports LHR JFK CDG

dagster-dev:
	dagster dev

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# Security
security-scan:
	bandit -r src/ -f json -o bandit-report.json
	safety check

# Performance
performance-test:
	pytest tests/ -m performance -v --benchmark-only

# Database
db-setup:
	python -c "from src.utils.config import Config; c = Config(); print('Database setup completed')"

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# All checks (for CI)
all-checks: format-check lint type-check test-cov security-scan

# Quick development setup
dev-setup: install-dev pre-commit-install db-setup

# Production setup
prod-setup: install docker-build

# Clean and rebuild everything
rebuild: clean install-dev build

# Show project info
info:
	@echo "Flight Operations Optimiser"
	@echo "Version: $(shell python -c 'import src; print(src.__version__)')"
	@echo "Python: $(shell python --version)"
	@echo "Location: $(shell pwd)" 

# Dataset preprocessing
filter-dataset: ## Filter and sample the flights dataset for development
	@echo "🔍 Filtering flights dataset for development..."
	@python scripts/test_dataset_filter.py

create-dev-sample: ## Create a development sample (50K rows)
	@echo "📊 Creating development sample..."
	@python -c "from src.data.preprocessing.dataset_filter import DatasetFilter; \
		filter_tool = DatasetFilter(); \
		df = filter_tool.generate_filtered_dataset('data/flights_dev_sample.csv', sample_size=50000); \
		print(f'Created development sample: {len(df)} rows')"

create-airline-sample: ## Create airline-specific sample (100K rows, major airlines)
	@echo "✈️ Creating airline sample..."
	@python -c "from src.data.preprocessing.dataset_filter import DatasetFilter; \
		filter_tool = DatasetFilter(); \
		df = filter_tool.generate_filtered_dataset('data/flights_airlines.csv', \
		airline_codes=['AA', 'DL', 'UA', 'WN'], sample_size=100000); \
		print(f'Created airline sample: {len(df)} rows')"

create-recent-sample: ## Create recent data sample (75K rows, last 3 months of available data)
	@echo "📅 Creating recent data sample..."
	@python -c "from src.data.preprocessing.dataset_filter import DatasetFilter; \
		filter_tool = DatasetFilter(); \
		df = filter_tool.generate_filtered_dataset('data/flights_recent.csv', \
		date_range=('2023-06-01', '2023-08-31'), sample_size=75000); \
		print(f'Created recent sample: {len(df)} rows')" 