# HPC DAG Scheduler Research Baseline

## 1. PROJECT SCOPE & ARCHITECTURE
We are building a research baseline to study optimal task scheduling of mathematical Directed Acyclic Graphs (DAGs) across an HPC cluster. 
- `/engine`: C++ MPI application. **STRICTLY READ-ONLY FOR CLAUDE.** Humans write this.
- `/hpcctl`: Python CLI (built with Typer/Click & `uv`) to manage AWS ParallelCluster lifecycles.
- `/tasks`: Python library (built with `uv`) containing the DAG math builder, node operations, and the actual task definition scripts.
- `/shared`: The serialization contract (JSON/Protobuf) linking `/tasks` to `/engine`.

## 2. STRICT PYTHON ENGINEERING STANDARDS
- **Package Manager:** Use `uv` exclusively for dependency management, environments, and running scripts (`uv run`, `uv add`).
- **Code Quality:** Use `ruff` for linting/formatting and `mypy --strict` for type checking. All functions MUST have Google-style docstrings.
- **Testing:** Use `pytest`. Test coverage must be high.
- **No Side Effects:** AWS commands MUST support a `--dry-run` flag that prints intended payloads instead of executing them.

## 3. EXCEPTION HANDLING PROTOCOL (CRITICAL)
We use a "Lazy Evaluation" architecture similar to Polars. The Python `/tasks` builder MUST catch all logical errors before the engine ever sees the DAG.
**Python (`/tasks`) MUST raise custom exceptions for:**
- `ShapeMismatchError`: Matrix dimensions do not align for the specific mathematical operation (e.g., N x M dot M x P).
- `DimensionalityError`: Operation called on wrong tensor rank (e.g., cross product on 2D matrix).
- `CyclicDependencyError`: Graph is not a valid DAG.
- `UninitializedNodeError`: Missing PRNG seeds or shape definitions for `init` nodes.
**C++ (`/engine`) handles ONLY runtime physics:**
- Out of Memory (OOM), MPI deadlocks, Schema parsing failures, Slurm Preemption, NaN/Inf mathematical anomalies.

## 4. DOCUMENTATION & SECRETS
- Update `README.md` iteratively with clear setup instructions, architecture diagrams (mermaid), and the core research problem statement.
- **NEVER** hardcode AWS credentials, SSH keys, or cluster IP addresses. Use environment variables. Ensure `.gitignore` aggressively hides `.env`, `*.json` configs, and `*.pem` keys.