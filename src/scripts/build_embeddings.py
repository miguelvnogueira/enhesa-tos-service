import logging
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup structural logging output formats
logging.basicConfig(
    format="%(asctime)s.%(msecs)d %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Resolve project pathing topologies (moves two steps out from src/scripts/ to find root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.info(f"Project root added to execution runtime path: {PROJECT_ROOT}")

from src.core.dataset_parser import parse_claudette_zipfile
from src.core.feature_engine import SimpleEmbeddingEngine
from src.core.classifier import ClauseClassifier

def main():
    zip_filepath = PROJECT_ROOT / "data" / "ToS.zip"
    logging.info(f"Fetching raw data from {zip_path}& parsing...")

    labeled_sentences_df=parse_claudette_zipfile(zip_filepath)
    
    logging.info("Initializing the Semantic Vector Space Engine...")
    engine = SimpleEmbeddingEngine(model_name="all-MiniLM-L6-v2")

    # 1. Compute the dense text-coordinate matrices
    X_embeddings = engine.build_index(labeled_sentences_df)
    logging.info(f"Successfully generated text embeddings matrix of shape: {X_embeddings.shape}")

    artifact_dir = PROJECT_ROOT / "models" / "search_artifacts"
    logging.info(f"Saving compiled matrices and metadata definitions to {artifact_dir}...")
    engine.save_artifacts(artifact_dir)

    logging.info(f"Processing complete! Embedding binaries and indexes are safely frozen on disk.")








if __name__ == "__main__":
    main()