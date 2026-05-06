# MSMolcL — Mass Spectrometry Molecular Retrieval Platform

A chemical deep-model integration platform for mass spectrometry data. Users upload MGF spectrum files and JSON fragmentation trees, and the system runs GPU-accelerated retrieval against PubChem or custom libraries to identify compounds.

## Architecture

```
web/
├── backend/                     # Python FastAPI server (:6008)
│   ├── main.py                  # App entry point, lifespan, CORS, static mounts
│   ├── api/                     # Route handlers (auth, upload, retrieve, history, etc.)
│   ├── core/                    # config.py, db.py, auth.py, rate_limit.py, memory_store.py
│   ├── models/                  # SQLAlchemy ORM (User)
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── services/                # Business logic: preprocessing, retrieval, Sirius workers
│   │   └── model/               # PyTorch models (GNN, encoders), training scripts
│   └── util/                    # Path helpers, task directory utilities
├── frontend/                    # Vue 3 + Vite SPA (:6006)
│   └── src/
│       ├── api/                 # Axios API wrappers
│       ├── components/          # ChemSearch, AdvancedSearch, SpectrumDetailCard, etc.
│       ├── views/               # Route pages: Login, TaskDetail, HistoryRecords, Docs
│       ├── utils/               # storage.js (auth/session), statasBatch.js
│       └── router/              # Vue Router config
└── .gitignore
```

### Data directory (sibling to `web/`)

```
autodl-tmp/                      # NOT in this repo — download separately
├── *.pth                        # PyTorch model weights
├── pubchem_final_with_formula.parquet
└── database/                    # Per-library parquet files in subdirectories
```

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **npm** 9+
- **CUDA-capable GPU** (required for retrieval inference)

## Quick Start

### 1. Clone and set up the backend

```bash
cd web

# Create virtual environment
conda create -n msmolcl python=3.10
conda activate msmolcl

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Set up the frontend

```bash
cd frontend
npm install
```

### 3. Prepare model and database files

Create a sibling directory `autodl-tmp/` next to `web/` and place the following files:

```
autodl-tmp/
├── model_epoch-36_vloss-0.1429.pth     # Positive-ion retrieval model
├── model_epoch-43_vloss-0.2149.pth     # Negative-ion retrieval model
├── ft_e8_loss1.0037.pth                # Advanced retrieval (positive)
├── ft_e5_loss1.0547.pth                # Advanced retrieval (negative)
├── pubchem_final_with_formula.parquet  # PubChem compound database
└── database/                           # Custom library databases
    ├── pubchem/
    ├── ymdb/
    └── ...
```

> **Note:** Model and database files are large (PubChem parquet ~20 GB). Ensure sufficient disk space.

### 4. Configuration

Key settings are in `backend/core/config.py`. Sensitive values can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET_KEY` | (built-in default) | Secret key for JWT token signing |
| `DATABASE_URL` | `sqlite:///.../users.db` | SQLAlchemy database URL |
| `CORS_ALLOW_ORIGINS` | localhost origins | Comma-separated list of allowed origins |
| `CORS_ALLOW_ORIGIN_REGEX` | (empty) | Regex for dynamic origin matching |
| `BACKEND_HOST` | `0.0.0.0` | Backend listen address |
| `BACKEND_PORT` | `6008` | Backend listen port |
| `SIRIUS_CELERY_BROKER_URL` | `redis://127.0.0.1:6379/0` | Sirius worker broker |

### 5. Run

**Backend:**
```bash
conda activate msmolcl
cd backend
python main.py
# Starts on http://0.0.0.0:6008
```

**Frontend (development):**
```bash
cd frontend
npm run dev
# Starts on http://127.0.0.1:6006, proxies /api-backend → :6008
```

**Frontend (production build):**
```bash
cd frontend
npm run build
# Output in dist/ — served by the backend at :6008
```

## Upload & Retrieval Flow

1. **Upload** MGF spectrum files (with optional JSON fragmentation trees) via the web UI
2. **MGF-only mode:** Files are sent to Sirius workers (Celery) to generate fragmentation trees asynchronously
3. **Select candidates** from PubChem or custom libraries
4. **Run retrieval** — GPU workers process spectra against the selected database and return ranked compound matches
5. **View results** — detailed spectrum cards with molecular structure visualization

## Storage

- **User accounts:** SQLite via SQLAlchemy (`users.db`)
- **History:** Per-user JSON files under `user_data/<username>/`
- **Task files:** Per-task directories under `user_data/<username>/task_<task_id>/`
- **Session state:** Browser `sessionStorage` (prefixed `u:<username>:`)

## License

This project is provided for research and educational purposes.
