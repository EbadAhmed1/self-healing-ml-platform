# Self-Healing ML Serving Platform

> **Portfolio project** demonstrating production ML infrastructure — automated
> drift detection, scheduled retraining, canary deployment routing, an LLM-powered diagnosis agent, and multi-tenant serving — built on FastAPI, PostgreSQL, scikit-learn, and Streamlit.

---

## Architecture Diagram

```
                             ┌──────────────────────────────────────┐
                             │    GitHub Actions Scheduled Cron     │
                             │ (drift_check.yml / retrain.yml / CI) │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
┌──────────────────┐               ┌────────────────────────┐               ┌───────────────────┐
│ Traffic Simulator│──────────────►│ Render Web Service     │──────────────►│ Hugging Face Hub  │
│ (Local/Production│  HTTP Predict │ (FastAPI Serving App)  │  HF Artifacts │ (Model Registry)  │
└──────────────────┘               └────────────┬───────────┘               └───────────────────┘
                                                │
                                                ├─────────────────────────┐
                                                ▼                         ▼
                                   ┌────────────────────────┐  ┌────────────────────┐
                                   │ Supabase / Postgres DB │  │ Slack Webhook      │
                                   │ (Predictions/Outcomes/ │  │ (Incident Alerts)  │
                                   │ Incidents/Deployments) │  └────────────────────┘
                                   └────────────┬───────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │ Streamlit Cloud        │
                                   │ (Monitoring Dashboard) │
                                   └────────────────────────┘
```

---

## Production Cloud Deployment Stack

- **Serving API & Diagnosis Agent**: Render (Free Web Service Tier via Docker)
- **Database**: PostgreSQL (Supabase Free Tier)
- **Model Registry**: Hugging Face Hub (Public/Private Versioned Model Repository)
- **Scheduled Monitoring & Retraining Jobs**: GitHub Actions Cron Workflows
- **Monitoring Dashboard**: Streamlit Community Cloud
- **Incident Alerts**: Slack / Discord Webhook Notifications

---

## Proving Generalization & Multi-Tenancy

The self-healing platform was designed from day one to be **multi-tenant**, where `model_id` (`{model_name}:{version}`) is a first-class field across all database tables, monitoring jobs, and diagnosis agent queries.

To prove that the platform generalizes without hardcoding or requiring parallel database tables, we added a **second tenant** (`fraud-model` — Transaction Fraud Detection) alongside the original `churn-model`.

### 100% Reused Infrastructure (Zero Code Modifications Required)

1. **Database Schema**: 100% shared Postgres tables (`predictions`, `outcomes`, `drift_reports`, `accuracy_reports`, `alerts`, `deployments`, `incidents`, `llm_usage`). Predictions store input feature vectors in a JSONB column (`input_features`), enabling different schemas per tenant in the same table.
2. **Monitoring & PSI Engine (`monitoring/`)**: Calculates feature drift and rolling accuracy strictly filtered by `model_id`.
3. **Diagnosis Agent (`agent/`)**: Evidence gathering, priority rule engine, and LLM reasoning layer execute seamlessly for `fraud-model` alerts without any rule changes.
4. **Remediation & Retraining (`models/retrain.py`, `models/canary_manager.py`)**: Gated retraining and 10% canary traffic promotion/rollback operate per `model_id`.
5. **Traffic Simulator (`simulator/`)**: Uses shared failure injection modes (`--drift-feature`, `--corrupt-feature`) for both tenants.

---

## Project Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **0** | Project scaffolding & Docker Compose | ✅ Done |
| **1** | Serving infrastructure & baseline churn model | ✅ Done |
| **2** | Traffic simulator & failure injection engine | ✅ Done |
| **3** | PSI drift detection & rolling accuracy monitoring | ✅ Done |
| **4** | Priority rule-based diagnosis & auto-remediation agent | ✅ Done |
| **5** | LLM reasoning layer for escalated incidents | ✅ Done |
| **6** | Gated production retraining & canary deployment | ✅ Done |
| **7** | Multi-tenant architecture & second model (`fraud-model`) | ✅ Done |
| **8** | Streamlit monitoring dashboard | ✅ Done |
| **9** | Production cloud deployment & GitHub Actions CI/CD | ✅ Done |

---

## Folder Structure

```
self-healing-ml-platform/
│
├── app/                    # FastAPI serving application
│   ├── routers/
│   │   ├── churn.py        #   POST /predict/churn-model (Tenant #1)
│   │   └── fraud.py        #   POST /predict/fraud-model (Tenant #2)
│   ├── database.py         #   SQLAlchemy engine with connection pooling
│   └── model_loader.py     #   Multi-model registry loader & HF Hub cache
│
├── models/                 # Training scripts + model utilities
│   ├── feature_config.py   #   Churn feature definitions
│   ├── feature_config_fraud.py # Fraud feature definitions
│   ├── pipeline_shared.py  #   Shared pipeline builder & snapshot engine
│   ├── train.py            #   Churn model training script
│   ├── train_fraud.py      #   Fraud model training script
│   ├── retrain.py          #   Gated production retrainer
│   └── canary_manager.py   #   Canary promotion & alert rollback
│
├── monitoring/             # Drift detection and scheduled metric jobs
│   ├── psi.py              #   PSI engine built from scratch
│   ├── check_drift.py      #   Feature & prediction drift checker
│   └── check_accuracy.py   #   Rolling accuracy tracker
│
├── agent/                  # Autonomous diagnosis & remediation agent
│   ├── evidence.py         #   Cross-table telemetry evidence gatherer
│   ├── diagnosis.py        #   Priority rule hypothesis engine
│   ├── remediation.py      #   Confidence-gated rollback engine
│   ├── llm.py              #   Groq LLM explanation layer
│   └── notifications.py    #   Slack/Discord webhook alert dispatcher
│
├── dashboard/              # Streamlit monitoring application
│   ├── app.py              #   Interactive multi-tenant dashboard
│   └── queries.py          #   Cached, parameterized SQLAlchemy queries
│
├── simulator/              # Synthetic traffic generator with failure injection
├── tests/                  # Complete pytest suite (74+ tests)
├── alembic/                # Database migration environment (0001–0007)
├── .github/workflows/      # GitHub Actions (ci.yml, drift_check.yml, retrain.yml)
├── Dockerfile              # Production multi-stage Docker build
├── render.yaml             # Render cloud deployment specification
└── README.md
```

---

## Deployment & Production Setup

### Step 1 — Database Setup (Supabase / Managed Postgres)
1. Create a free PostgreSQL database instance on [Supabase](https://supabase.com/).
2. Copy the Connection DSN string (Format: `postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres`).
3. Run Alembic migrations against the production database:
   ```bash
   export DATABASE_URL="postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres"
   alembic upgrade head
   ```

### Step 2 — Model Artifact Storage (Hugging Face Hub)
1. Create a public or private repository on [Hugging Face Hub](https://huggingface.co/new) (e.g. `your-username/self-healing-ml-models`).
2. Set environment variables:
   ```bash
   export USE_HF_HUB=true
   export HF_REPO_ID="your-username/self-healing-ml-models"
   export HF_HUB_TOKEN="hf_..."
   ```

### Step 3 — FastAPI Serving App (Render Web Service)
1. Fork or push this repository to GitHub.
2. Log into [Render](https://render.com/) and create a new **Web Service** using Docker.
3. Link your repository and set the following environment secrets on Render:
   - `DATABASE_URL`: Your Supabase connection string.
   - `GROQ_API_KEY`: Groq API Key for LLM reasoning.
   - `SLACK_WEBHOOK_URL`: Slack Incoming Webhook URL.
   - `USE_HF_HUB`: `false` (or `true` if using Hugging Face Hub).
4. Deploy the service. Once deployed, verify API liveness at `https://<your-app>.onrender.com/health`.

### Step 4 — Streamlit Monitoring Dashboard (Streamlit Community Cloud)
1. Log into [Streamlit Cloud](https://streamlit.io/cloud) and deploy `dashboard/app.py`.
2. In Advanced Settings -> Secrets, add:
   ```toml
   [postgres]
   DATABASE_URL = "postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres"
   ```

### Step 5 — Scheduled Jobs (GitHub Actions)
1. In your GitHub repository settings under **Secrets and variables -> Actions**, add:
   - `DATABASE_URL`: Production Supabase Postgres connection string.
   - `GROQ_API_KEY`: Groq API Key for LLM explanations.
   - `SLACK_WEBHOOK_URL`: Slack Incoming Webhook URL.
2. The `.github/workflows/drift_check.yml` (every 6h) and `retrain.yml` (every 12h) will automatically execute on schedule against production DB.

---

## Environment Variables & Secrets Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | SQLAlchemy connection DSN for Postgres DB |
| `MODEL_REGISTRY_PATH` | No | Local directory path to model artifacts (Default: `./models/artifacts`) |
| `USE_HF_HUB` | No | Set `true` to fetch model artifacts from Hugging Face Hub |
| `HF_REPO_ID` | Optional | Hugging Face repository ID (`username/repo-name`) |
| `HF_HUB_TOKEN` | Optional | Hugging Face Access Token |
| `GROQ_API_KEY` | Optional | API key for Groq LLM diagnosis explanations |
| `SLACK_WEBHOOK_URL` | Optional | Incoming webhook URL for Slack alert notifications |
| `API_BASE_URL` | No | Base URL of deployed FastAPI app / dashboard link |
| `LOG_LEVEL` | No | Python logging level (`INFO`, `DEBUG`) |

---

## Free-Tier Architectural Constraints & Trade-offs

This portfolio project is designed to run entirely on free-tier infrastructure. The following design trade-offs and mitigations are implemented:

1. **Render Free-Tier Cold Starts**: Render puts free web services to sleep after 15 minutes of inactivity. The initial prediction request after sleep may take ~30 seconds to spin up container instances. *In enterprise production, dedicated compute instances or Kubernetes pods with minimum replica counts would be used.*
2. **Supabase Postgres Connection Limits**: Free-tier PostgreSQL limits total simultaneous client connections (~15–20 max). The application configures SQLAlchemy connection pooling (`pool_size=5`, `max_overflow=10`, `pool_recycle=1800`, `pool_pre_ping=True`) in `app/database.py` to prevent connection exhaustion under simulator traffic.
3. **GitHub Actions Runner Minute Allocation**: Free GitHub accounts receive 2,000 Action runner minutes per month. Scheduled workflows are set to 6-hour (`drift_check.yml`) and 12-hour (`retrain.yml`) cron intervals, consuming ~150 runner minutes per month.

---

## Local Development & Testing Guide

### Prerequisites
- Python 3.12 (or 3.11)
- Docker Desktop (optional for local Postgres)

### Step 1 — Clone and setup virtual environment
```bash
git clone https://github.com/<your-username>/self-healing-ml-platform.git
cd self-healing-ml-platform
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt -r requirements-dev.txt
```

### Step 2 — Run full pytest test suite
```bash
pytest tests/ -v
```

### Step 3 — Run traffic simulator against local or production API
```bash
# Local API simulation
python simulator/run.py --api-url http://localhost:8000 --model churn-model

# Production Render API simulation
python simulator/run.py --api-url https://<your-app>.onrender.com --model churn-model
```

### Step 4 — Launch local Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
