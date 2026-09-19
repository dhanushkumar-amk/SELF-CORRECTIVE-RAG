# ============================================================================
#  Self-Correcting RAG — Developer Convenience Script (Windows)
# ============================================================================
#  Usage: .\dev.ps1 <command>
#  Run without arguments to see available commands.
# ============================================================================

param(
    [Parameter(Position=0)]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Show-Help {
    Write-Host ""
    Write-Host "  Self-Correcting RAG — Developer Commands" -ForegroundColor Cyan
    Write-Host "  =========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  .\dev.ps1 setup          " -NoNewline -ForegroundColor Green
    Write-Host "Install all dependencies (backend + frontend)"
    Write-Host "  .\dev.ps1 setup-backend  " -NoNewline -ForegroundColor Green
    Write-Host "Set up Python venv and install backend deps"
    Write-Host "  .\dev.ps1 setup-frontend " -NoNewline -ForegroundColor Green
    Write-Host "Install frontend npm packages"
    Write-Host "  .\dev.ps1 backend        " -NoNewline -ForegroundColor Green
    Write-Host "Start FastAPI dev server (port 8000)"
    Write-Host "  .\dev.ps1 frontend       " -NoNewline -ForegroundColor Green
    Write-Host "Start Next.js dev server (port 3000)"
    Write-Host "  .\dev.ps1 test           " -NoNewline -ForegroundColor Green
    Write-Host "Run backend pytest suite"
    Write-Host "  .\dev.ps1 docker-up      " -NoNewline -ForegroundColor Green
    Write-Host "Build and start via Docker Compose"
    Write-Host "  .\dev.ps1 docker-down    " -NoNewline -ForegroundColor Green
    Write-Host "Stop Docker Compose services"
    Write-Host "  .\dev.ps1 clean          " -NoNewline -ForegroundColor Green
    Write-Host "Remove .venv, node_modules, caches"
    Write-Host ""
}

function Invoke-SetupBackend {
    Write-Host "`n  Setting up backend..." -ForegroundColor Yellow
    Push-Location "$ProjectRoot\backend"
    try {
        if (-not (Test-Path ".venv")) {
            python -m venv .venv
        }
        .venv\Scripts\pip.exe install --upgrade pip | Out-Null
        .venv\Scripts\pip.exe install -r requirements.txt
        if (-not (Test-Path ".env")) {
            Copy-Item .env.example .env
            Write-Host "  Created .env from .env.example" -ForegroundColor DarkGray
        }
        Write-Host "  ✓ Backend dependencies installed" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

function Invoke-SetupFrontend {
    Write-Host "`n  Setting up frontend..." -ForegroundColor Yellow
    Push-Location "$ProjectRoot\frontend"
    try {
        npm install
        if (-not (Test-Path ".env.local")) {
            Copy-Item .env.local.example .env.local
            Write-Host "  Created .env.local from .env.local.example" -ForegroundColor DarkGray
        }
        Write-Host "  ✓ Frontend dependencies installed" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

switch ($Command) {
    "setup" {
        Invoke-SetupBackend
        Invoke-SetupFrontend
        Write-Host "`n  ✓ Setup complete! Run '.\dev.ps1 backend' and '.\dev.ps1 frontend' to start." -ForegroundColor Green
        Write-Host ""
    }
    "setup-backend" {
        Invoke-SetupBackend
    }
    "setup-frontend" {
        Invoke-SetupFrontend
    }
    "backend" {
        Push-Location "$ProjectRoot\backend"
        try {
            .venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
        } finally {
            Pop-Location
        }
    }
    "frontend" {
        Push-Location "$ProjectRoot\frontend"
        try {
            npm run dev
        } finally {
            Pop-Location
        }
    }
    "test" {
        Push-Location "$ProjectRoot\backend"
        try {
            .venv\Scripts\python.exe -m pytest tests/ -v
        } finally {
            Pop-Location
        }
    }
    "docker-up" {
        docker-compose up --build
    }
    "docker-down" {
        docker-compose down
    }
    "clean" {
        $dirs = @(
            "backend\.venv", "backend\.pytest_cache", "backend\__pycache__",
            "frontend\node_modules", "frontend\.next"
        )
        foreach ($d in $dirs) {
            $path = Join-Path $ProjectRoot $d
            if (Test-Path $path) {
                Remove-Item $path -Recurse -Force
                Write-Host "  Removed $d" -ForegroundColor DarkGray
            }
        }
        Write-Host "  ✓ Cleaned up" -ForegroundColor Green
    }
    default {
        Show-Help
    }
}
