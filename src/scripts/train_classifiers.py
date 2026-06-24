import logging
import joblib
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

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
from src.core.classifier import ClauseClassifier,





def main():

    logging.info("Instantiate a fresh engine instance")
    search_engine = SimpleEmbeddingEngine()

    artifacts_dir = PROJECT_ROOT / "models" / "search_artifacts"
    logging.info(f"Loading precomputed embeddings from {artifacts_dir}...")
    search_engine.load_artifacts(artifacts_dir)

    logging.info(f"Getting predictor variables -> embeddings...")
    X = search_engine.corpus_embeddings
    logging.info(f"Getting target variable 'is_unfair'...")
    y = search_engine.metadata_df['is_unfair'].values



    # Establish explicit directory mapping
    models_output_dir = PROJECT_ROOT / "models" / "classifiers"
    models_output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = {}

    for model_name, estimator in MODELS_TO_EXPERIMENT.items():
        logging.info(f"🚀 Starting benchmark for: {model_name}")
        
        # Instantiate the wrapper
        clause_clf = ClauseClassifier(estimator=estimator)
        
        # 1. FIX: Assign to the specific model key so we don't overwrite the dict
        metrics_payload = clause_clf.train_and_evaluate(X, y, test_size=0.2, random_state=42)
        report_dict[model_name] = metrics_payload
        
        # Define unique subdirectory path for this model
        model_save_path = models_output_dir / model_name
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 2. Save the model binary using your class tool (creates 'clf_estimator.joblib')
        logging.info(f"💾 Freezing weights for {model_name}...")
        clause_clf.save_model(model_save_path)
        
        # 3. Save the specific evaluation report dict inside the same folder
        report_file_path = model_save_path / "evaluation_report.joblib"
        logging.info(f"📊 Saving evaluation report for {model_name} at: {report_file_path}")
        joblib.dump(metrics_payload, report_file_path)

    logging.info("🎉 All models and their respective reports are safely frozen to disk!")

if __name__ == "__main__":
    main()