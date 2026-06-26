# enhesa-tos-service

A production-grade Python service implementing semantic similarity search and legal compliance risk classification on the CLAUDETTE corpus. Built with clean modular architecture, rigorous experimental methodology, and robust data alignment pipelines. The decoupled core engine features comprehensive REST API endpoints and a containerized runtime environment.

---

## Context

Most consumers accept online Terms of Service (ToS) contracts without reading them, creating a severe informational asymmetry that online platforms frequently exploit by hiding legally unfair or detrimental clauses. While European consumer law prohibits these practices, consumer protection agencies lack the labor and resources to manually audit thousands of text-dense documents.  To solve this bottleneck, this project builds a containerized, production-ready backend service that utilizes semantic search and machine learning (ML) classification, trained on the CLAUDETTE corpus dataset (Lippi et al., 2019). It automatically detects potentially unfair clauses at a sentence level without relying on massive, high-latency Large Language Models (LLMs).  

---

## Technical Grounding & Literature Context

To establish enterprise credibility and defensibility , the architectural decisions implemented across this codebase are firmly grounded in recent peer-reviewed scientific literature:  
- Embedding Superiority: Moving beyond the original baseline bag-of-words (BoW) and TF-IDF models established by Lippi et al. (2019) (which topped out at a macro F1 of 0.806) , this architecture embraces dense text vectors.  
- Domain-Specific Advantage: Following Akash et al. (2024), we map clauses to a specialized Legal-BERT space, which dramatically outperforms general-purpose models (like all-MiniLM-L6-v2) by successfully capturing deep semantic nuances and avoiding vocabulary fragmentation on complex contractual terminology.  
- Classifier Optimization: While a broad experimental matrix was evaluated (spanning Logistic Regression, LinearSVC, Random Forests, Histogram Gradient Boosting, and MLPs) , our testing validated that pairing Legal-BERT with an RBF-Kernel Support Vector Classifier (SVC RBF) yields the strongest standalone performance topology, nearly matching state-of-the-art benchmarks (Macro F1 of 0.871).

---

## System Architecture & Data Fl

graph TD
    %% Define Styles and Colors
    classDef default fill:#1e1e2e,stroke:#cdd6f4,stroke-width:1px,color:#cdd6f4;
    classDef input fill:#313244,stroke:#f5e0dc,stroke-width:2px,color:#f5e0dc;
    classDef processing fill:#11111b,stroke:#89b4fa,stroke-width:1px,color:#89b4fa;
    classDef flow fill:#181825,stroke:#a6e3a1,stroke-width:1px,color:#a6e3a1;
    classDef output fill:#313244,stroke:#f38ba8,stroke-width:2px,color:#f38ba8;

    %% Workflow Nodes
    A[Incoming Target Clause] :::input --> B[Legal-BERT Vectorization]:::processing
    B --> C[Dense Embedding Vector<br>768-Dim float32]:::processing
    
    %% Dual Flows
    C -->|Route Vector| D[Global Classification Flow<br>Calibrated Classifier SVC-RBF<br>Platt Scaling Curve]:::flow
    C -->|Query Vector| E[Local Semantic Neighborhood Flow<br>Spatial Index Vector Search<br>Pulls Top-K K=5 Closest Elements]:::flow
    
    %% Merge at Bayesian Layer
    D -->|Continuous Prior Probability| F[Bayesian Post-Processor]:::processing
    E -->|Continuous Similarity Scores| F
    
    %% Final Outputs
    F --> G[Context-Aware Posterior Probability]:::processing
    G -->|Decision Threshold >= 0.50| H[JSON Payload Delivery<br>0/1 Verdict + Top-K Neighbors]:::output


    
                  ┌──────────────────────┐
                  │    Raw ToS Input     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Text Ingestion Layer │
                  │  (List/String Coerce)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Legal-BERT Space   │
                  │ (768-Dim float32)    │
                  └─────┬──────────┬─────┘
                        │          │
        ┌───────────────┘          └───────────────┐
        ▼                                          ▼
┌───────────────┐                          ┌───────────────┐
│ Spatial Index │                          │Ensemble Matrix│
│ (Cosine Sim)  │                          │ (Frozen Joblib│
└───────┬───────┘                          │  Estimators)  │
        │                                  └───────┬───────┘
        │                                          │
        └───────────────┐          ┌───────────────┘
                        ▼          ▼
                  ┌──────────────────────┐
                  │ JSON Payload Delivery│
                  │(0/1 Verdict + Top-K) │
                  └──────────────────────┘


For more details, the user is referred to the Report.pdf.

---


## Getting Started & Run Guidelines
### Prerequisites
Docker and Docker Compose installed locally.

Note for Linux users: Ensure your user is appended to the docker group, or prefix operational commands with sudo.

### Step 1. Spin up the Containerized API
Run the orchestration layout command to build your environment and boot the REST interface.

For Standard (macOS, Windows, or configured Linux):
```
docker build --no-cache -t enhesa-tos-api .
```

The Linux fallback (if daemon socket permissions are not configured):
```
sudo docker build --no-cache -t enhesa-tos-api .
```

### Step 2. Fire up the Containers
For Standard (macOS, Windows, or configured Linux):
```
docker-compose up
```

The Linux fallback (if daemon socket permissions are not configured):
```
sudo docker-compose up
```

### Step 3. Send a test request
curl -X POST http://localhost:8000/api/v1/analyze \
     -H "Content-Type: application/json" \
     -d '{"sentence": "The provider reserves the right to modify these terms at any time without prior notification.", "top_k":3}'







# RUNNING UI

Start your backend server
```
poetry run uvicorn src.api.main:app --reload --port 8000
```

Start frontend app
```
poetry run streamlit run ui/app.py
```