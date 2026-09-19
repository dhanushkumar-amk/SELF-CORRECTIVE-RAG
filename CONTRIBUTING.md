# Engineering Standards & Contribution Guide

This guide defines the git workflow, branching model, commit message conventions, and pull request process for the **Self-Correcting RAG with Hallucination Detection** project.

Even as a solo project, following these lightweight engineering practices ensures that the 50-phase architecture remains clean, maintainable, and legible to technical interviewers and code reviewers.

---

## 1. Branching Strategy

### The `main` Branch
- `main` represents the **always deployable, production-ready source of truth**.
- **Rule:** Never commit or push directly to `main` from Phase 5 onward.
- All changes must land on `main` through Pull Requests.

### Phase Branches
Work is broken down into structured phases. Each phase has its own dedicated feature branch following this exact naming format:

```text
phase-<number>-<short-description>
```

**Examples:**
- `phase-05-git-workflow`
- `phase-06-pdf-upload-endpoint`
- `phase-10-embedding-generation`
- `phase-16-dense-retrieval`
- `phase-30-nli-hallucination-verification`

### Standard Workflow per Phase
```bash
# 1. Start from an up-to-date main branch
git checkout main
git pull origin main

# 2. Create the branch for the new phase
git checkout -b phase-06-pdf-upload-endpoint

# 3. Implement and test locally
pytest backend/tests/ -v
cd frontend && npm run lint && npm run build

# 4. Commit changes following Conventional Commits (enforced by git hook)
git add -A
git commit -m "feat(ingestion): add multipart PDF upload endpoint with schema validation"

# 5. Push phase branch to GitHub
git push -u origin phase-06-pdf-upload-endpoint

# 6. Open a Pull Request on GitHub using the PR template
# 7. Confirm CI passes and merge into main!
```

---

## 2. Commit Message Conventions (Conventional Commits)

Commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <short imperative description>
```

### Allowed Types
| Type | Purpose | Example |
|---|---|---|
| `feat` | New feature or capability | `feat(retrieval): add BM25 sparse index build and ranking` |
| `fix` | Bug fix or error resolution | `fix(ingestion): handle corrupted PDF stream during text extraction` |
| `test` | Adding or updating tests | `test(config): add fail-fast validation and placeholder detection tests` |
| `docs` | Documentation updates | `docs(readme): add step-by-step pinecone index creation guide` |
| `refactor`| Code changes without feature/fix changes | `refactor(pinecone): consolidate client initialization into singleton` |
| `chore` | Tooling, config, dependencies | `chore(deps): update pinecone client to v10.0.0` |
| `ci` | CI/CD pipelines & GitHub Actions | `ci(github): add automated pytest and next-build workflows` |
| `perf` | Performance improvements | `perf(retrieval): cache MiniLM tokenizer during batch encoding` |

### Key Rules
- Use lower-case imperative verbs (e.g. `add`, `fix`, `update`, not `added` or `adds`).
- Do not end the subject line with a period.
- Scope is optional but encouraged (e.g. `(retrieval)`, `(config)`, `(frontend)`).

> **Note on Commit History:**  
> Initial commits from Phases 1–4 were created during repo scaffolding and setup. Conventional Commits and automated git hook validation are enforced strictly **from Phase 5 onward**.

---

## 3. Git Hooks & Automated Safeguards

This repository uses automated hooks via `pre-commit`:
1. **Secret Scanning (`detect-secrets`):** Blocks commits containing API keys, private tokens, or secrets.
2. **Commit Message Validation (`commit-msg`):** Automatically validates that every commit message follows Conventional Commits formatting before allowing the commit.

### Developer Setup (One-time)
```bash
# Install hooks into local .git directory
pip install pre-commit detect-secrets
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## 4. Pull Request & Review Process

When opening a Pull Request on GitHub, the [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) will populate automatically.

### PR Requirements
1. **Title:** Use Conventional Commits format (e.g., `feat(phase-05): git workflow, PR templates, and CI setup`).
2. **Body:** Complete the Phase number, summary of changes, and verification proof.
3. **CI Status:** The automated GitHub Actions CI workflow must pass:
   - Backend pytest suite
   - Frontend linting and production build
4. **Merge Strategy:** Use **Squash and merge** or standard **Merge commit** to preserve phase milestones in git history.

---

## 5. Branch Protection on `main` (Manual GitHub Setup)

To protect `main` from accidental direct pushes and guarantee every phase passes through a Pull Request, configure branch protection in GitHub:

1. Open your repository on GitHub: [`https://github.com/dhanushkumar-amk/SELF-CORRECTIVE-RAG`](https://github.com/dhanushkumar-amk/SELF-CORRECTIVE-RAG)
2. Click the **Settings** tab in the top navigation bar.
3. In the left-hand sidebar under **Code and automation**, click **Branches**.
4. In the **Branch protection rules** section, click **Add branch protection rule** (or **Add rule**).
5. In the **Branch name pattern** field, enter: `main`.
6. Under **Protect matching branches**, configure:
   - [x] **Require a pull request before merging**
     - *(Solo tip: You can uncheck "Require approvals" so you don't need a second account to approve, while still enforcing the PR flow)*
   - [x] **Require status checks to pass before merging**
     - Search for and select: `Backend Tests & Lint` and `Frontend Lint & Build`.
   - [x] **Do not allow bypassing the above settings** (optional, recommended).
7. Click **Create** (or **Save changes** at the bottom).
