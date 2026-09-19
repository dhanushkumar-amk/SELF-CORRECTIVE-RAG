# Self-Correcting RAG — Frontend

Next.js frontend for the Self-Correcting RAG with Hallucination Detection system.

## Prerequisites

- **Node.js 18+** (developed with Node 24.20.0)
- npm 9+

> **nvm users:** A `.nvmrc` file is included — run `nvm use` to switch.

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Copy environment config
cp .env.local.example .env.local
# Edit if backend runs on a non-default URL

# 3. Start the development server
npm run dev
```

App runs at **http://localhost:3000**.

## Environment Variables

Copy `.env.local.example` to `.env.local`:

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

## Tech Stack

- **Next.js 16** (App Router)
- **TypeScript**
- **Tailwind CSS v4**
- **shadcn/ui** (component library — Phase 42+)

## Project Structure

```
app/
├── layout.tsx         # Root layout with Geist fonts
├── page.tsx           # Landing page with backend health check
├── globals.css        # Tailwind imports and CSS variables
└── favicon.ico
```
