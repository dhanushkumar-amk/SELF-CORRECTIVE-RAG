# ============================================================================
#  Self-Correcting RAG — Developer Convenience Commands
# ============================================================================
#  Usage: make <target>
#  Run `make help` to see all available targets.
# ============================================================================

.PHONY: help setup setup-backend setup-frontend backend frontend test \
        docker-up docker-down clean

# ── Default ──────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo.
	@echo   Self-Correcting RAG — Available Commands
	@echo   ════════════════════════════════════════════
	@echo.
	@echo   make setup          Install all dependencies (backend + frontend)
	@echo   make backend        Start FastAPI dev server (port 8000)
	@echo   make frontend       Start Next.js dev server (port 3000)
	@echo   make test           Run backend pytest suite
	@echo   make docker-up      Build and start all services via Docker Compose
	@echo   make docker-down    Stop Docker Compose services
	@echo   make clean          Remove .venv, node_modules, caches
	@echo.

# ── Setup ────────────────────────────────────────────────────────────────────

setup: setup-backend setup-frontend ## Install all dependencies
	@echo.
	@echo   ✓ Setup complete. Run 'make backend' and 'make frontend' to start.
	@echo.

setup-backend: ## Set up Python venv and install backend deps
	cd backend && python -m venv .venv
	cd backend && .venv\Scripts\pip.exe install --upgrade pip
	cd backend && .venv\Scripts\pip.exe install -r requirements.txt
	@if not exist backend\.env ( copy backend\.env.example backend\.env )
	@echo   ✓ Backend dependencies installed

setup-frontend: ## Install frontend npm packages
	cd frontend && npm install
	@if not exist frontend\.env.local ( copy frontend\.env.local.example frontend\.env.local )
	@echo   ✓ Frontend dependencies installed

# ── Development ──────────────────────────────────────────────────────────────

backend: ## Start FastAPI dev server with hot reload
	cd backend && .venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000

frontend: ## Start Next.js dev server
	cd frontend && npm run dev

# ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run backend test suite
	cd backend && .venv\Scripts\python.exe -m pytest tests/ -v

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up: ## Build and start all services
	docker-compose up --build

docker-down: ## Stop all services
	docker-compose down

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove generated files and caches
	if exist backend\.venv rmdir /s /q backend\.venv
	if exist backend\.pytest_cache rmdir /s /q backend\.pytest_cache
	if exist frontend\node_modules rmdir /s /q frontend\node_modules
	if exist frontend\.next rmdir /s /q frontend\.next
	@echo   ✓ Cleaned up
