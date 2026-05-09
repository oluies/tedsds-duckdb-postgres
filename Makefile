.PHONY: help install pg-up pg-down migrate test lint typecheck fmt clean

PG_DSN ?= postgresql://tedsds:tedsds@localhost:5432/tedsds

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python deps via uv
	uv sync

pg-up: ## Start local Postgres in docker
	docker compose up -d postgres
	@echo "Waiting for Postgres..."
	@until docker compose exec -T postgres pg_isready -U tedsds >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres ready on localhost:5432"

pg-down: ## Stop local Postgres
	docker compose down

migrate: ## Apply SQL migrations
	@for f in migrations/*.sql; do \
		echo "Applying $$f"; \
		docker compose exec -T postgres psql -U tedsds -d tedsds -v ON_ERROR_STOP=1 -f - < $$f; \
	done

test: ## Run tests
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Format with ruff
	uv run ruff format .

typecheck: ## Static type check with mypy
	uv run mypy

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name __pycache__ -prune -exec rm -rf {} +
