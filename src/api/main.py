import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.core.feature_engine import LegalBertEmbeddingEngine
from src.core.classifier import ClauseClassifier
from src.core.clause_processor import process_user_sentence
from src.api.schemas import AnalysisRequest, AnalysisResponse, MatchResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for loading heavy ML artifacts."""
    try:
        logger.info("Initializing embedding engine and models...")
        
        artifacts_exp = "legal_bert_embeddings"
        artifacts_dir = PROJECT_ROOT / "models" / artifacts_exp
        search_engine = LegalBertEmbeddingEngine()
        search_engine.load_artifacts(artifacts_dir)
        ml_models["search_engine"] = search_engine
        
        clause_clf = ClauseClassifier()
        mlp_model_dir = artifacts_dir / "classifiers" / "SVC_RBF_Pipeline"
        clause_clf.load_model(mlp_model_dir)
        ml_models["clause_clf"] = clause_clf
        
        logger.info("Artifacts successfully loaded into memory.")
        yield
    except Exception as e:
        logger.error(f"Critical error during startup artifact initialization: {e}")
        raise e
    finally:
        # Clean up resources if necessary
        ml_models.clear()

app = FastAPI(
    title="Legal Clause Fairness Analysis Service",
    description="A semantic similarity search and compliance classification engine for Terms of Service.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["Infrastructure"])
def health_check():
    """Simple health check endpoint for monitoring/k8s probes."""
    if not ml_models.get("search_engine") or not ml_models.get("clause_clf"):
        raise HTTPException(status_code=503, detail="Models are not ready")
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_clause(payload: AnalysisRequest):
    """
    Analyzes a user-provided clause sentence.
    1. Runs semantic search against historical data.
    2. Runs classification to flag potentially unfair terms.
    """
    try:
        search_engine = ml_models["search_engine"]
        clause_clf = ml_models["clause_clf"]
        
        # Call your existing domain processing logic
        search_results, is_unfair = process_user_sentence(
            search_engine, 
            clause_clf, 
            payload.sentence, 
            payload.top_k
        )
        
        formatted_matches = []
        for rank, match in enumerate(search_results, 1):
            formatted_matches.append(
                MatchResult(
                    rank=rank,
                    similarity_score=float(match.get('similarity_score', 0.0)),
                    company=match.get('company', 'Unknown'),
                    text=match.get('text', ''),
                    historical_ground_truth='Unfair' if match.get('is_unfair') == 1 else 'Fair'
                )
            )
            
        return AnalysisResponse(
            sentence=payload.sentence,
            is_unfair=bool(is_unfair == 1),
            status_label="UNFAIR CLAUSE" if is_unfair == 1 else "COMPLIANT CLAUSE",
            top_matches=formatted_matches
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis engine error.")