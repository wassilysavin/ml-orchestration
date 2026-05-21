# Decoupled stack — running each component as its own job over network services

This is **step 1 of the "each component on its own node" ladder**: the same Task 3
flows, but coordinating through *network services* (an MLflow tracking server and
an S3-compatible object store) instead of a shared local filesystem. It runs as
multiple containers on one host; moving to true multi-node is then a scheduler
swap (Argo/Kubeflow/Nomad), not a code change.

## What changes vs. the local runs

The local flows coordinate through three bind-mounted directories — `data/`,
`models/`, `mlruns/`. Here:

| Local coupling | Replaced by |
|---|---|
| `mlruns/` file backend | **MLflow tracking server** (HTTP) + **Postgres** backend store |
| `models/<run_id>/` local read | **MinIO** (S3) artifact store; the A/B job pulls the model from MLflow by `run_id` |
| `data/` | read-only dataset bind mount (the one input still on disk; point at object storage to remove it) |

The flow jobs share **no** `models/` or `mlruns/` directory. Training jobs log
their model to MLflow → MinIO; the A/B job resolves a `flow_version_id` via the
server and pulls the model artifact back. That is the exact cross-node handoff a
shared local disk was hiding.

This is driven entirely by one environment variable: when `MLFLOW_TRACKING_URI`
is set, the code (`src/mlflow_setup.py`) talks to the server and
`registry.load_model` falls back to the MLflow artifact store when a model is not
on local disk. Unset, every flow behaves exactly as the local runs do — same
code, no branches in the flow logic.

## Prerequisites

* Docker + Docker Compose v2.
* The processed dataset present on the host at `src/pipeline/data/processed/`
  (it is mounted read-only). If missing, generate it once locally:
  `python -m src.prepare_dataset` from `src/pipeline/`.

## Run it

```bash
cd deploy
cp .env.example .env

# 1) bring up the shared services
docker compose up -d                      # postgres + minio + mlflow server

# 2) run each component as its own one-shot job
docker compose run --rm train-baseline    # → flow version dfda8c64c3e3
docker compose run --rm train-challenger   # → flow version e39940d83d56
docker compose run --rm monitor            # drift test on the unseen segment
docker compose run --rm abtest             # A/B across the two flow versions

# 3) inspect
open http://localhost:5000                 # MLflow UI: 3 experiments
open http://localhost:9001                 # MinIO console: artifacts under mlflow/

# teardown (add -v to drop the postgres/minio volumes)
docker compose down
```

Each `run --rm` is a separate container — i.e. a separate "node" — and they only
ever talk to each other through the MLflow server and MinIO.

## Verified

The decoupled code path was exercised end-to-end against a real MLflow server
(HTTP tracking + proxied artifact store), not just the local file backend:

* both training flows logged runs + model artifacts to the server;
* with the local `models/` directory removed, the A/B flow resolved each
  `flow_version_id` over HTTP and **pulled both ONNX models from the artifact
  store** before predicting (A=0.880 vs B=0.802);
* the monitoring flow logged its drift run to the server.

So the only thing standing between this and components on genuinely separate
nodes is the executor: replace `docker compose run` with pods scheduled by
Argo/Kubeflow/Nomad, and point `data/` at object storage. The flow logic, the
`flow_version_id` / `ab_test_id` contracts, and the step boundaries are unchanged.

## Notes

* **Proxied artifacts.** The server runs with `--serve-artifacts`, so flow
  clients need only `MLFLOW_TRACKING_URI`; the server holds the MinIO
  credentials and brokers artifact upload/download. Drop `--serve-artifacts` to
  have clients talk to MinIO directly (they already carry `boto3` + `AWS_*`).
* **Robustness step** is non-fatal here for the same reason as the local flow:
  the model is persisted before robustness runs, and a deliberately weaker
  challenger must still be produced for the A/B comparison.
