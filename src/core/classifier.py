import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



MODELS_TO_EXPERIMENT = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", 
        max_iter=1000, 
        random_state=42
    ),
    "LinearSVC": LinearSVC(
        class_weight="balanced", 
        dual="auto", 
        max_iter=5000, 
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced", 
        n_estimators=200, 
        random_state=42, 
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    ),
    "MLPClassifier":MLPClassifier(
        # Defines a DNN with 3 hidden layers: 128 neurons -> 64 neurons -> 32 neurons
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,  # Automatically stops training if validation score plateaus to prevent overfitting
        validation_fraction=0.1
    )
}



class ClauseClassifier:
    """
    Handles training, evaluation, and serialization of modular machine learning 
    classifiers trained over dense semantic embedding feature spaces.
    """
    
    def __init__(self, estimator=None) -> None:
        """
        Initializes the classifier wrapper with a pluggable scikit-learn estimator.

        Parameters:
        -----------
        estimator : scikit-learn estimator, optional
            Any initialized scikit-learn classifier instance (e.g., LogisticRegression(), 
            RandomForestClassifier()). Defaults to LogisticRegression(max_iter=1000).
        """
        if estimator is None:
            # High max_iter guarantees convergence on deep embedding matrices
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
            logging.info("No estimator specified. Defaulting to Balanced Logistic Regression.")
        else:
            self.model = estimator
            logging.info(f"Initialized wrapper with custom estimator: {estimator.__class__.__name__}")

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Splits dataset, fits the integrated estimator, and prints out a scientific breakdown 
        of core tracking metrics.

        Parameters:
        -----------
        X : np.ndarray
            The 2D dense embedding feature array matrix of shape (N, 384).
        y : np.ndarray
            The 1D binary target array tracking unfair classifications.

        Returns:
        --------
        dict
            Dictionary containing core test evaluation scores.
        """
        logging.info(f"Splitting dataset (Test size: {test_size * 100}%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logging.info(f"Fitting {self.model.__class__.__name__} to train vectors...")
        self.model.fit(X_train, y_train)
        
        logging.info("Executing evaluation inference predictions...")
        predictions = self.model.predict(X_test)
                
        return classification_report(y_test, predictions, output_dict=True)

    def predict(self, query_embedding: np.ndarray) -> int:
        """Runs binary classification inference against raw vector spaces."""
        if self.model is None:
            raise ValueError("Classifier states are uninitialized. Train or load a model first.")
        return int(self.model.predict(query_embedding)[0])

    def save_model(self, output_dir: Path) -> None:
        """Serializes the fitted estimator architecture to disk storage."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "clf_estimator.joblib"
        joblib.dump(self.model, model_path)
        logging.info(f"Model estimator successfully frozen at: {model_path}")

    def load_model(self, output_dir: Path) -> None:
        """Loads a pre-compiled structural model framework back into RAM memory lanes."""
        model_path = output_dir / "clf_estimator.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No serial structures found at target pathway: {model_path}")
        self.model = joblib.load(model_path)
        logging.info(f"Model framework successfully restored from {model_path}")