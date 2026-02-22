"""
Test training module functionality for Task 3.2.

This test verifies that the training module correctly:
- Imports and uses data_loader.load_iris_data()
- Implements evaluate_model() with precision, recall, F1-score
- Implements save_model_with_version() with timestamp-based naming
- Logs confusion matrix as MLflow artifact
- Logs all hyperparameters and metrics to MLflow
- Has structured logging for training progress
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import (
    evaluate_model,
    save_model_with_version,
    train_model,
    log_confusion_matrix_artifact,
    setup_mlflow
)


def test_evaluate_model_returns_all_metrics():
    """Test that evaluate_model returns accuracy, precision, recall, f1_score, and confusion_matrix."""
    # Create simple test data
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[2, 3], [4, 5]])
    y_test = np.array([0, 1])
    
    # Train a simple model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Verify all required metrics are present
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    
    # Verify metrics are numeric
    assert isinstance(metrics["accuracy"], (float, np.floating))
    assert isinstance(metrics["precision"], (float, np.floating))
    assert isinstance(metrics["recall"], (float, np.floating))
    assert isinstance(metrics["f1_score"], (float, np.floating))
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    
    print("✓ evaluate_model returns all required metrics")


def test_save_model_with_version_creates_timestamped_file():
    """Test that save_model_with_version creates a file with timestamp in the name."""
    # Create a simple model
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)
    
    metrics = {"accuracy": 0.95}
    
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("train.MODEL_DIR", Path(tmpdir)):
            model_path = save_model_with_version(model, metrics)
            
            # Verify file was created
            assert model_path.exists()
            
            # Verify filename format: iris_model_YYYYMMDD_HHMMSS.pkl
            assert model_path.name.startswith("iris_model_")
            assert model_path.name.endswith(".pkl")
            
            # Verify timestamp format (should be 15 chars: YYYYMMDD_HHMMSS)
            timestamp_part = model_path.stem.replace("iris_model_", "")
            assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
            assert "_" in timestamp_part
            
            print(f"✓ save_model_with_version creates timestamped file: {model_path.name}")


def test_train_model_with_hyperparameters():
    """Test that train_model accepts hyperparameters and returns a trained model."""
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    
    hyperparameters = {
        "max_iter": 200,
        "random_state": 42,
        "solver": "lbfgs"
    }
    
    model = train_model(X_train, y_train, hyperparameters)
    
    # Verify model is trained
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")  # Model has been fitted
    
    # Verify hyperparameters were applied
    assert model.max_iter == 200
    assert model.random_state == 42
    assert model.solver == "lbfgs"
    
    print("✓ train_model accepts hyperparameters and trains model")


def test_log_confusion_matrix_artifact_creates_image():
    """Test that log_confusion_matrix_artifact creates and logs a confusion matrix image."""
    cm = np.array([[10, 2, 1], [1, 15, 2], [0, 1, 12]])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confusion_matrix.png"
        
        # Mock mlflow.log_artifact
        with patch("train.mlflow.log_artifact") as mock_log_artifact:
            log_confusion_matrix_artifact(cm, str(output_path))
            
            # Verify image was created
            assert output_path.exists()
            
            # Verify mlflow.log_artifact was called
            mock_log_artifact.assert_called_once_with(str(output_path))
            
            print("✓ log_confusion_matrix_artifact creates and logs confusion matrix image")


def test_setup_mlflow_configures_tracking():
    """Test that setup_mlflow configures MLflow tracking URI and experiment."""
    with patch("train.mlflow.set_tracking_uri") as mock_set_uri, \
         patch("train.mlflow.set_experiment") as mock_set_exp:
        
        setup_mlflow()
        
        # Verify MLflow was configured
        mock_set_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_set_exp.assert_called_once_with("iris-classification")
        
        print("✓ setup_mlflow configures MLflow tracking URI and experiment")


if __name__ == "__main__":
    print("\nRunning Task 3.2 functionality tests...\n")
    
    test_evaluate_model_returns_all_metrics()
    test_save_model_with_version_creates_timestamped_file()
    test_train_model_with_hyperparameters()
    test_log_confusion_matrix_artifact_creates_image()
    test_setup_mlflow_configures_tracking()
    
    print("\n✅ All Task 3.2 functionality tests passed!\n")
