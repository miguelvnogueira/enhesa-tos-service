# enhesa-tos-service
A production-grade Python service implementing semantic similarity search and legal compliance risk classification on the CLAUDETTE corpus. Built with clean modular architecture, rigorous experimental methodology, and robust data alignment pipelines. Decoupled core engine features comprehensive REST API endpoints and a containerized runtime.


# Context
Most consumers accept online Terms of Service (ToS) contracts without reading them, creating a severe informational asymmetry that online platforms frequently exploit by hiding legally unfair or detrimental clauses. While European consumer law prohibits these practices, consumer protection agencies lack the labor and resources to manually audit thousands of text-dense documents. To solve this bottleneck, this project builds a containerized, production-ready backend service that utilizes semantic search and machine learning (ML) classification, trained on the CLAUDETTE corpus dataset (Lippi et al, 2019), to automatically detect and categorize potentially unfair clauses at a sentence level without relying on large language models (LLMs).


# Technical Details
## How to Run the Service

### Prerequisites
- Docker and Docker Compose installed.
- *Note for Linux users:* Ensure your user is added to the `docker` group, or prefix commands with `sudo`.

### Step 1: Spin up the Containerized API
Run the orchestration command. If you encounter a socket permission error on Linux, use the `sudo` variant:

For Standard (macOS, Windows, or configured Linux):
```
docker-compose up --build
```

The Linux fallback (if daemon permissions are not configured):
```
sudo docker-compose up --build
```