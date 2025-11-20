# LLM Experiment Platform - Design Document

**Date:** 2025-11-20
**Status:** Approved Design
**Target Users:** Individual researchers and ML engineering teams
**Deployment:** Cloud-native SaaS on Google Cloud Platform

## Overview

A comprehensive LLM experiment platform with LLMOps capabilities, designed to help researchers and teams run, track, and evaluate LLM experiments. The platform integrates with Weights & Biases (W&B) as the tracking backend and provides Git-based prompt management, multi-provider support, and automated evaluation orchestration.

## Design Principles

- **Modular monolith architecture** - Simple deployment, clear module boundaries, scalable
- **Cloud-native on GCP** - Fully managed services, minimal operational overhead
- **W&B as tracking backend** - Leverage existing tools rather than rebuilding
- **Git-based prompt management** - Version control and collaboration via standard Git workflows
- **Provider agnostic** - Start with OpenAI and Anthropic, extensible to others
- **Evaluation orchestration** - Integrate with external evaluation tools
- **Python-first** - Accessible to ML engineers, rich ecosystem

---

## 1. High-Level Architecture

The platform is a Python-based modular monolith with these core layers:

### Web Layer
- **FastAPI application** serving REST APIs for programmatic access
- **Streamlit application** providing the web UI (runs as separate process, shares backend modules)
- Both communicate with the same underlying service modules

### Service Layer (Core business logic modules)
- `experiments`: Experiment lifecycle management (create, configure, submit, monitor)
- `prompts`: Git-based prompt repository management and versioning
- `providers`: LLM provider integrations (OpenAI, Anthropic, + extensible)
- `evaluations`: Evaluation orchestration and integration with external tools
- `jobs`: Job queue management and worker coordination
- `tracking`: W&B integration for metrics and experiment tracking
- `auth`: User authentication and session management

### Infrastructure Layer (GCP-native)
- **Cloud SQL for PostgreSQL**: Managed PostgreSQL for experiments metadata, user data, job status, provider configs
- **Memorystore for Redis**: Managed Redis for Celery job queue, caching, real-time updates
- **Cloud Storage (GCS)**: Object storage for large artifacts (datasets, model outputs, logs)
- **Git repositories**: Prompt templates and versions (GitHub, GitLab, Cloud Source Repositories)
- **W&B**: Experiment metrics, visualizations, and run tracking

### Compute (GCP deployment)
- **Cloud Run**: Host FastAPI API and Streamlit UI (containerized, auto-scaling)
- **Celery workers**: Can run on Cloud Run Jobs, GCE instances, or GKE depending on workload patterns

### Execution Layer
- **Celery workers**: Pull jobs from Redis queue, execute LLM calls, run evaluations
- Workers can scale horizontally based on queue depth

---

## 2. Core Data Models

Primary entities in PostgreSQL:

### User
- Basic auth info (email, hashed password, OAuth tokens)
- W&B API key (for pushing metrics)
- LLM provider API keys (encrypted, per-user quotas)
- Preferences and settings

### Experiment
- Metadata: name, description, creator, timestamps
- Configuration: model provider, model name, hyperparameters (temperature, max_tokens, etc.)
- Prompt reference: Git repo URL, branch, commit SHA, file path
- Dataset reference: GCS path or inline test cases
- Evaluation config: which evaluators to run, success criteria
- Status: draft, queued, running, completed, failed
- W&B run ID: link to W&B for metrics visualization

### Job
- Links to parent Experiment
- Job type: llm_inference, evaluation, bulk_run
- Status: pending, running, completed, failed, cancelled
- Worker info: worker ID, start time, end time
- Result summary: success count, failure count, cost
- Error logs if failed

### PromptRepository
- Git repository URL (GitHub, GitLab, etc.)
- Access credentials (SSH key, OAuth token)
- Last synced commit SHA
- User/team ownership

### ProviderConfig
- Provider name (openai, anthropic, etc.)
- API endpoint overrides (for custom deployments)
- Rate limits and quotas
- Cost tracking settings

### EvaluationRun
- Links to Experiment and Job
- Evaluator type (w&b_weave, custom_function, etc.)
- Results summary: pass/fail counts, aggregate scores
- GCS path to detailed results

**Relationships:** Users create Experiments → Experiments spawn Jobs → Jobs execute LLM calls and trigger EvaluationRuns. PromptRepositories are cloned locally by workers when executing jobs.

---

## 3. Experiment Execution Flow

### 1. Experiment Definition (UI or API)
- User creates experiment via Streamlit UI or FastAPI endpoint
- Specifies: prompt repo/file, model provider & config, test dataset, evaluators
- Platform validates: prompt file exists in repo, API keys present, dataset accessible
- Experiment saved as "draft" in PostgreSQL

### 2. Experiment Submission
- User clicks "Run" → experiment status changes to "queued"
- Platform creates W&B run via SDK, gets run ID, stores in Experiment record
- Creates Job record in PostgreSQL with type "llm_inference"
- Publishes job to Celery queue (Redis)

### 3. Job Execution (Celery Worker)
- Worker pulls job from queue, updates Job status to "running"
- Clones/pulls prompt repository to local temp directory
- Loads prompt template from specified file path (at specific commit SHA)
- Loads dataset from GCS or inline data
- For each test case:
  - Renders prompt with test case variables
  - Calls LLM provider API (with retry logic, rate limiting)
  - Logs request/response to W&B (tokens, latency, cost)
  - Stores outputs temporarily
- Updates W&B run with summary metrics (total tokens, cost, latency percentiles)

### 4. Evaluation (if configured)
- Worker creates EvaluationRun record
- Triggers configured evaluators with LLM outputs
- For W&B Weave: uses W&B SDK to run evaluations
- For custom evaluators: executes user-defined Python functions
- Logs evaluation results to W&B and PostgreSQL
- Saves detailed results JSON to GCS

### 5. Completion
- Worker marks Job as "completed" (or "failed" if errors)
- Updates Experiment status to "completed"
- Final W&B run sync
- User can view results in UI or W&B directly

---

## 4. Module Structure & Responsibilities

### `platform/api/` - FastAPI application
- `routes/experiments.py`: CRUD endpoints for experiments
- `routes/jobs.py`: Job status, cancellation, logs
- `routes/prompts.py`: List repos, browse files, view diffs
- `routes/auth.py`: Login, signup, token refresh
- `dependencies.py`: Auth middleware, DB sessions, rate limiting

### `platform/ui/` - Streamlit application
- `pages/experiments.py`: Create/view experiments, visual experiment builder
- `pages/results.py`: Results dashboard with W&B embedded views
- `pages/prompts.py`: Browse prompt repos, view versions
- `pages/settings.py`: API keys, W&B integration, provider configs
- `components/`: Reusable Streamlit components

### `platform/services/` - Core business logic
- `experiment_service.py`: Experiment CRUD, validation, submission
- `prompt_service.py`: Git operations (clone, pull, checkout, diff)
- `provider_service.py`: LLM provider clients with unified interface
- `evaluation_service.py`: Orchestrate evaluators, parse results
- `tracking_service.py`: W&B SDK wrapper, logging helpers
- `job_service.py`: Job lifecycle management

### `platform/workers/` - Celery workers
- `tasks.py`: Celery task definitions (run_experiment, run_evaluation)
- `executor.py`: LLM call execution with retry/rate limiting
- `evaluators/`: Evaluation integrations (weave.py, custom.py)

### `platform/models/` - SQLAlchemy ORM models
- Database table definitions (User, Experiment, Job, etc.)

### `platform/core/` - Shared utilities
- `config.py`: Environment config, secrets management
- `database.py`: DB connection, session management
- `storage.py`: GCS client wrapper
- `security.py`: Password hashing, token generation, encryption

---

## 5. Error Handling & Reliability

### Job-Level Resilience
- **Automatic retries**: Celery tasks retry on transient failures (network errors, rate limits) with exponential backoff
- **Timeout handling**: Jobs have configurable max runtime, auto-cancel if exceeded
- **Partial completion**: If a job processes 100 test cases and fails on #87, save the first 86 results before failing
- **Dead letter queue**: Failed jobs after max retries go to DLQ for manual inspection

### Provider Reliability
- **Circuit breakers**: If a provider (e.g., OpenAI) is consistently failing, temporarily pause jobs using that provider
- **Fallback providers**: Optional - if primary provider fails, automatically try alternate (OpenAI → Anthropic with same prompt)
- **Rate limit handling**: Respect provider rate limits, queue requests if hitting limits
- **Cost safety limits**: Hard stop if job exceeds user-defined cost threshold

### Data Integrity
- **Atomic operations**: Experiment status updates and job creation happen in database transactions
- **Idempotent tasks**: Jobs can be safely retried without duplicate work (check if outputs already exist)
- **Audit logging**: All state changes logged to PostgreSQL for debugging and compliance

### Monitoring & Alerts
- **Health checks**: API and workers expose health endpoints for Cloud Run monitoring
- **Metrics export**: Job queue depth, success/failure rates, provider latency to Cloud Monitoring
- **User notifications**: Email/webhook alerts when jobs complete or fail (optional)
- **Cost alerts**: Notify users when approaching budget limits

### Recovery Mechanisms
- **Worker crashes**: Jobs marked as "running" but worker died get auto-requeued after timeout
- **Database backups**: Cloud SQL automated backups for disaster recovery
- **Graceful shutdown**: Workers finish current task before shutdown (Cloud Run termination handling)

---

## 6. Prompt Management (Git-Based)

### Repository Registration
- Users register Git repositories via UI or API (GitHub, GitLab, Bitbucket, Cloud Source Repositories)
- Platform stores: repo URL, authentication credentials (SSH key or OAuth token), default branch
- On registration, platform does initial clone to validate access and list available prompt files

### Prompt File Structure
- Prompts stored as files in the repo (`.txt`, `.md`, or `.jinja2` for templating)
- Support for Jinja2 templating: `{{ variable }}` placeholders filled at runtime
- Metadata in YAML frontmatter or separate `.meta.yaml` files:
```yaml
name: "Customer Support Classifier"
description: "Classifies customer support tickets"
variables: [customer_message, context]
model_suggestions: [gpt-4, claude-3-opus]
```

### Versioning & Selection
- When creating an experiment, user selects: repository → branch/tag → specific file
- Platform records the exact commit SHA at selection time (not just branch name)
- Workers checkout that specific SHA when executing, ensuring reproducibility
- Users can compare prompt versions using Git diff in the UI

### Workflow Integration
- **Browse**: UI shows file tree of registered repos, preview prompt content
- **Edit**: Users edit prompts in their own Git workflow (local editor, GitHub web editor, etc.)
- **Sync**: Platform polls repos periodically or accepts webhooks to detect new commits
- **Rollback**: Can create new experiment using older commit SHA if needed

### Collaboration Benefits
- Teams use standard Git workflows (branches, PRs, code review for prompts)
- Prompt changes tracked alongside code changes in version control
- Full audit trail of who changed what and when
- No custom prompt editor needed - leverage existing Git tools

---

## 7. LLM Provider Abstraction

### Provider Interface
All providers implement a common interface:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, config: dict) -> LLMResponse:
        """Execute a completion request"""

    @abstractmethod
    def stream_complete(self, prompt: str, config: dict) -> Iterator[str]:
        """Stream completion chunks"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for cost estimation"""

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
```

### Provider Implementations
- `platform/providers/openai.py`: OpenAI client (GPT-4, GPT-3.5, etc.)
- `platform/providers/anthropic.py`: Anthropic client (Claude models)
- `platform/providers/factory.py`: Factory to instantiate correct provider based on config

### Unified Response Format
All providers return standardized `LLMResponse`:
```python
@dataclass
class LLMResponse:
    content: str  # The actual completion
    model: str  # Model used
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    provider_metadata: dict  # Provider-specific extras
```

### Configuration Mapping
- Platform uses standardized config keys that map to provider-specific parameters
- `temperature`, `max_tokens`, `top_p` map to all providers
- Provider-specific params can be passed in `extra_params` dict

### Rate Limiting & Retry
- Each provider implementation handles its own rate limits
- Shared retry decorator with exponential backoff
- Token bucket algorithm to stay under rate limits proactively

---

## 8. W&B Integration as Backend

### Initialization & Setup
- Users provide W&B API key in settings (stored encrypted)
- Platform creates W&B projects per user (e.g., `llm-experiments-{user_id}`)
- Each experiment becomes a W&B run with tags for filtering

### Run Creation & Tracking
When an experiment starts:
```python
run = wandb.init(
    project=f"llm-experiments-{user_id}",
    name=experiment.name,
    config={
        "model": experiment.model_name,
        "provider": experiment.provider,
        "temperature": experiment.temperature,
        "prompt_commit": experiment.prompt_commit_sha,
    }
)
# Store run.id in Experiment record
```

### Metrics Logged to W&B
Per test case:
- `input_tokens`, `output_tokens`: Token usage
- `latency_ms`: Response time
- `cost_usd`: Cost per request
- `prompt`, `completion`: Actual text

Aggregate metrics:
- `total_cost`, `total_tokens`, `avg_latency`
- `p50_latency`, `p95_latency`, `p99_latency`
- Evaluation scores (accuracy, custom metrics)

### Artifacts Storage
- Full prompt templates uploaded as W&B artifacts
- Large datasets referenced by GCS path (not duplicated to W&B)
- Evaluation results uploaded as JSON artifacts
- Model outputs logged as W&B Tables for browsing

### W&B UI Integration
- Streamlit UI embeds W&B charts and dashboards using iframe or W&B public URLs
- Users can click through to full W&B workspace for deep analysis
- Platform provides curated views but W&B is source of truth for metrics

### Fallback Consideration
- Basic metrics still stored in PostgreSQL (cost, tokens, status)
- W&B is best-effort for rich tracking, not critical path for job execution

---

## 9. Evaluation Orchestration

### Evaluation Configuration
Users specify evaluators when creating experiments:
```yaml
evaluators:
  - type: "w&b_weave"
    scorers: ["exact_match", "semantic_similarity"]
  - type: "custom_function"
    function_path: "my_repo/evaluators/custom_scorer.py"
    parameters:
      threshold: 0.8
```

### Built-in Evaluator Types

**1. W&B Weave Integration**
- Uses W&B Weave SDK to run evaluations
- Platform passes LLM outputs + expected outputs to Weave
- Weave runs scorers (exact match, LLM-as-judge, etc.)
- Results automatically logged to W&B run

**2. Custom Python Functions**
- Users provide Python function in their Git repo:
```python
def evaluate(output: str, expected: str, metadata: dict) -> dict:
    return {
        "score": 0.95,
        "passed": True,
        "details": "..."
    }
```
- Worker imports and executes function for each test case
- Platform logs results to W&B and PostgreSQL

**3. HTTP Webhook Evaluators** (future)
- POST LLM outputs to external evaluation service
- Useful for proprietary evaluation tools or custom APIs

### Evaluation Execution Flow
1. After LLM inference job completes, worker checks if evaluations configured
2. Creates `EvaluationRun` record for each evaluator
3. Executes evaluators sequentially or in parallel
4. Aggregates results (pass/fail counts, average scores)
5. Logs to W&B as evaluation metrics
6. Saves detailed results JSON to GCS

### Result Structure
Each evaluation produces:
- Per-test-case scores (W&B Table or GCS JSON)
- Aggregate metrics (avg score, pass rate, etc.)
- Pass/fail determination based on thresholds
- Comparison to baseline runs (if configured)

### UI Display
- Experiment results page shows evaluation summary
- Click through to see per-case breakdown
- Compare evaluation scores across experiment runs
- Filter/sort by evaluation results

---

## 10. Cost Tracking & Monitoring

### Real-Time Cost Tracking
- Each LLM API call calculates cost immediately using provider-specific pricing
- Costs stored at multiple levels:
  - Per API call (in W&B logs)
  - Per job (aggregated in Job table)
  - Per experiment (total cost in Experiment table)
  - Per user (running total for billing/quota)

### Pricing Configuration
`platform/providers/pricing.py` maintains current pricing:
```python
PRICING = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
    },
    "anthropic": {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015}
    }
}
```

### Budget Controls
Users can set limits:
- **Per-experiment budget**: Job auto-cancels if cost exceeds threshold
- **Daily/monthly quotas**: Prevent runaway costs across all experiments
- **Cost estimation**: Platform estimates cost before running based on test case count + avg tokens

### Cost Analytics Dashboard
Streamlit UI shows:
- Cost trends over time (daily/weekly/monthly)
- Cost breakdown by provider, model, experiment
- Most expensive experiments
- Cost per test case metrics
- Projected monthly spend

### Cost Optimization Features
- **Model recommendations**: Suggest cheaper models for similar tasks
- **Batch discounting**: Route eligible jobs through batch endpoints if available
- **Cache-aware execution**: Detect duplicate prompts, show potential savings

### Alerts & Notifications
- Email/webhook when approaching budget limits (80%, 100%)
- Weekly cost summary reports
- Anomaly detection: alert if costs spike unexpectedly

### Export & Reporting
- Export cost data to CSV for finance/accounting
- Integration with W&B for cost visualization alongside performance
- API endpoint for programmatic cost queries

---

## 11. Deployment & Infrastructure (GCP)

### Containerization
- Single Docker image containing all code (API, UI, workers)
- Different entry points for each service:
  - API: `uvicorn platform.api.main:app`
  - UI: `streamlit run platform/ui/app.py`
  - Worker: `celery -A platform.workers worker`
- Multi-stage Dockerfile to minimize image size

### GCP Services Setup

**Compute:**
- **Cloud Run (API)**: FastAPI service, auto-scales 0-N instances
- **Cloud Run (UI)**: Streamlit service, separate from API for independent scaling
- **Cloud Run Jobs (Workers)**: Celery workers as jobs, scale based on queue depth, OR
- **GCE Managed Instance Group**: Alternative for workers if long-running jobs need persistent connections

**Data & Storage:**
- **Cloud SQL (PostgreSQL)**: Primary database with automated backups, read replicas for analytics
- **Memorystore (Redis)**: Celery broker and result backend, 1-5GB instance
- **Cloud Storage**: Buckets for artifacts, datasets, logs

**Networking:**
- **Cloud Load Balancer**: Routes traffic to API and UI services
- **VPC**: Services communicate via private IPs, only load balancer exposed publicly
- **Cloud NAT**: Workers access external APIs (OpenAI, Anthropic) via NAT gateway

**Secrets Management:**
- **Secret Manager**: Store user API keys, database passwords, OAuth secrets
- Services access secrets via environment variables or direct API calls

**Monitoring & Logging:**
- **Cloud Logging**: Centralized logs from all services
- **Cloud Monitoring**: Metrics, dashboards, alerts for service health
- **Cloud Trace**: Distributed tracing for debugging slow requests
- **Error Reporting**: Automatic exception tracking

**Infrastructure as Code:**
- **Terraform** to define all GCP resources
- Separate environments: `dev`, `staging`, `prod`
- State stored in GCS backend

**CI/CD Pipeline:**
- **Cloud Build** triggered on git push
- Steps: lint → test → build Docker image → push to Artifact Registry → deploy to Cloud Run
- Automated database migrations on deploy

**Scaling Strategy:**
- API/UI: Cloud Run auto-scales based on request volume
- Workers: Scale based on Redis queue depth (Cloud Monitoring metric triggers instance group resize)
- Database: Start with single instance, add read replicas if needed, vertical scaling for writes

---

## 12. Security & API Design

### Authentication & Security

**API Authentication:**
- **JWT tokens**: Users login (email/password or OAuth), receive JWT access token
- **Token refresh**: Short-lived access tokens (15 min), refresh tokens for renewal
- **API keys**: Optional API keys for programmatic access (service accounts)
- All API endpoints require valid JWT in `Authorization: Bearer <token>` header

**Secrets Protection:**
- User API keys (OpenAI, Anthropic, W&B) encrypted at rest using GCP KMS
- Secrets never logged or returned in API responses
- Git credentials stored in Secret Manager, injected at runtime
- Database credentials rotated regularly

**Network Security:**
- HTTPS only (TLS 1.3), enforced by Cloud Load Balancer
- CORS configured for Streamlit UI origin only
- Rate limiting on API endpoints (per user): 100 req/min standard, 1000 req/min for batch operations
- SQL injection prevention via SQLAlchemy ORM (parameterized queries)

**Access Control:**
- Users can only access their own experiments, prompts, jobs
- Database queries filtered by `user_id` automatically (row-level security)
- Future team features: RBAC with team membership checks

### Core API Endpoints

**Authentication:**
- `POST /auth/register` - Create account
- `POST /auth/login` - Get JWT token
- `POST /auth/refresh` - Refresh token

**Experiments:**
- `POST /experiments` - Create experiment
- `GET /experiments` - List user's experiments (paginated, filterable)
- `GET /experiments/{id}` - Get experiment details
- `POST /experiments/{id}/run` - Submit experiment to queue
- `DELETE /experiments/{id}/cancel` - Cancel running experiment
- `GET /experiments/{id}/results` - Get results and evaluation scores

**Jobs:**
- `GET /jobs/{id}` - Job status and progress
- `GET /jobs/{id}/logs` - Stream job logs
- `POST /jobs/{id}/cancel` - Cancel job

**Prompts:**
- `POST /prompts/repos` - Register Git repository
- `GET /prompts/repos` - List registered repos
- `GET /prompts/repos/{id}/files` - Browse files in repo
- `GET /prompts/repos/{id}/file?path=...&commit=...` - Get file content at specific commit
- `POST /prompts/repos/{id}/sync` - Trigger repo sync

**Settings:**
- `PUT /settings/providers/{provider}` - Update provider API key
- `PUT /settings/wandb` - Update W&B configuration
- `GET /settings/usage` - Get cost and usage stats

All endpoints return JSON, follow REST conventions, include proper HTTP status codes and error messages.

---

## Next Steps

### Phase 1: Foundation (MVP)
1. Set up GCP infrastructure (Terraform)
2. Implement core data models and database schema
3. Build FastAPI endpoints for experiments and jobs
4. Implement OpenAI provider integration
5. Build basic Celery worker for job execution
6. Integrate W&B for experiment tracking
7. Create minimal Streamlit UI for experiment creation and viewing

### Phase 2: Prompt Management
1. Implement Git repository integration
2. Add prompt browsing and version selection UI
3. Build prompt rendering with Jinja2 templates
4. Add prompt comparison and diff viewing

### Phase 3: Evaluation & Cost Tracking
1. Implement W&B Weave evaluation integration
2. Add custom evaluator support
3. Build cost tracking and analytics
4. Create cost dashboard in UI
5. Add budget controls and alerts

### Phase 4: Polish & Scale
1. Add Anthropic provider integration
2. Implement error handling and retry logic
3. Add monitoring and alerting
4. Performance optimization
5. Documentation and user guides

### Phase 5: Team Features (Future)
1. Organization and team management
2. RBAC and permissions
3. Shared prompt repositories
4. Collaborative experiment workflows
