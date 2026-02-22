# Implementation Plan: MLOps Pipeline Fix

## Overview

This implementation plan addresses critical bugs in the existing iris classification MLOps pipeline and implements a complete, production-ready ML operations workflow. The tasks are organized to fix immediate issues first, then build out missing functionality, and finally integrate everything with comprehensive testing.

## Tasks

- [x] 1. Set up project structure and testing framework
  - Create `tests/` directory with `__init__.py`
  - Install and configure `pytest`, `hypothesis`, and `pytest-cov`
  - Create `conftest.py` with hypothesis profiles (dev: 10 examples, ci: 100 examples)
  - Set up logging configuration module
  - _Requirements: 6.3, 6.4, 6.5_

- [ ] 2. Implement data loading and validation module
  - [x] 2.1 Create `src/data_loader.py` with data loading functions
    - Implement `load_iris_data()` with CSV loading and sklearn fallback
    - Implement `validate_iris_dataframe()` with schema and content validation
    - Add logging for data source selection (CSV vs sklearn)
    - Handle empty/missing CSV files gracefully
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [ ]* 2.2 Write property test for data validation completeness
    - **Property 1: Data Validation Completeness**
    - **Validates: Requirements 1.2, 1.3, 1.4**
  
  - [ ]* 2.3 Write property test for validation error messages
    - **Property 2: Validation Error Descriptiveness**
    - **Validates: Requirements 1.5**
  
  - [ ]* 2.4 Write unit tests for data loader
    - Test loading from valid CSV
    - Test fallback to sklearn when CSV is empty
    - Test fallback to sklearn when CSV is missing
    - Test validation with various invalid schemas
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Fix and enhance training service
  - [x] 3.1 Refactor `src/train.py` to eliminate duplicate MLflow configuration
    - Create `setup_mlflow()` function for one-time configuration
    - Remove duplicate MLflow setup code (lines 11-12 and 35-36)
    - Add configuration constants at module level
    - _Requirements: 2.6_
  
  - [x] 3.2 Enhance training with comprehensive metrics and versioning
    - Import and use `data_loader.load_iris_data()`
    - Implement `evaluate_model()` function with precision, recall, F1-score
    - Implement `save_model_with_version()` with timestamp-based naming
    - Log confusion matrix as MLflow artifact
    - Log all hyperparameters and metrics to MLflow
    - Add structured logging for training progress
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2_
  
  - [ ]* 3.3 Write property test for training artifacts completeness
    - **Property 3: Training Artifacts Completeness**
    - **Validates: Requirements 2.2, 2.4, 8.2**
  
  - [ ]* 3.4 Write property test for model versioning consistency
    - **Property 4: Model Versioning Consistency**
    - **Validates: Requirements 8.1, 8.4**
  
  - [ ]* 3.5 Write unit tests for training service
    - Test model training with valid data
    - Test MLflow logging (mock MLflow server)
    - Test model saving with timestamp format
    - Test training failure handling and exit codes
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 8.1_

- [~] 4. Checkpoint - Ensure data loading and training tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement prediction module
  - [~] 5.1 Create `src/predict.py` with ModelPredictor class
    - Implement `__init__()` method with model and path attributes
    - Implement `load_latest_model()` to find and load most recent model by timestamp
    - Implement `predict()` method returning prediction, index, and confidence scores
    - Add error handling for missing/corrupted models
    - Add logging for model loading and predictions
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.3, 8.5_
  
  - [ ]* 5.2 Write property test for latest model selection
    - **Property 5: Latest Model Selection**
    - **Validates: Requirements 3.1, 8.3, 8.5**
  
  - [ ]* 5.3 Write property test for prediction response structure
    - **Property 6: Prediction Response Structure**
    - **Validates: Requirements 3.3, 3.4**
  
  - [ ]* 5.4 Write property test for input validation enforcement
    - **Property 7: Input Validation Enforcement**
    - **Validates: Requirements 3.2, 7.3**
  
  - [ ]* 5.5 Write unit tests for prediction module
    - Test loading latest model from multiple versions
    - Test prediction with valid input
    - Test prediction with invalid input (wrong number of features)
    - Test behavior when model is missing
    - Test behavior when model is corrupted
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 6. Fix FastAPI application structure and endpoints
  - [~] 6.1 Refactor `api/app.py` to fix indentation and structure
    - Move Pydantic models (IrisInput, PredictionResponse, HealthResponse) to module level
    - Move route handlers (@app.get, @app.post) to module level with proper decorators
    - Remove incorrect indentation inside IrisInput class
    - Add FastAPI metadata (title, description, version)
    - _Requirements: 4.1, 4.2_
  
  - [~] 6.2 Implement model loading and error handling in API
    - Import ModelPredictor from src.predict
    - Add startup event to load model on application start
    - Implement /health endpoint with model status
    - Implement /predict endpoint with proper error handling
    - Use HTTPException with correct status codes (400, 500, 503)
    - Add request/response models with Pydantic Field validation
    - Add logging for API requests and errors
    - _Requirements: 3.5, 4.3, 4.4, 4.5, 6.1, 6.2, 7.2, 7.3, 7.4_
  
  - [ ]* 6.3 Write property test for HTTP status code semantics
    - **Property 8: HTTP Status Code Semantics**
    - **Validates: Requirements 4.5**
  
  - [ ]* 6.4 Write unit tests for API endpoints
    - Test /predict endpoint with valid request (200 response)
    - Test /predict endpoint with invalid request (400 response)
    - Test /predict endpoint when model unavailable (503 response)
    - Test /health endpoint when model is loaded
    - Test /health endpoint when model is not loaded
    - Test proper HTTP status codes for various scenarios
    - _Requirements: 4.3, 4.4, 4.5, 6.1, 6.2, 7.2, 7.3_

- [~] 7. Checkpoint - Ensure prediction and API tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Fix Docker configuration
  - [~] 8.1 Refactor `api/Dockerfile` to fix path references
    - Remove `../` path references
    - Use relative paths from build context root
    - Copy `requirements.txt` from project root
    - Copy `src/` directory for imports
    - Copy `api/` directory for application code
    - Copy `models/` directory for model files
    - _Requirements: 5.1, 5.2_
  
  - [~] 8.2 Enhance Dockerfile with health checks and proper configuration
    - Add HEALTHCHECK directive calling /health endpoint
    - Expose port 8000
    - Use `--no-cache-dir` flag for pip install
    - Set proper CMD with uvicorn configuration
    - _Requirements: 5.3, 5.4, 5.5, 6.1, 6.2_
  
  - [ ]* 8.3 Write integration test for Docker container
    - Test container builds successfully
    - Test container starts and responds to health checks
    - Test /predict endpoint is accessible from container
    - _Requirements: 5.2, 5.3, 5.4, 5.5_

- [ ] 9. Implement comprehensive error handling and logging
  - [~] 9.1 Add error handling throughout the pipeline
    - Ensure all file I/O operations have try-except blocks
    - Ensure all API endpoints return appropriate error responses
    - Ensure training script exits with non-zero code on failure
    - Add descriptive error messages without exposing system internals
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ]* 9.2 Write property test for logging level appropriateness
    - **Property 9: Logging Level Appropriateness**
    - **Validates: Requirements 6.3, 6.4, 6.5**
  
  - [ ]* 9.3 Write property test for graceful error handling
    - **Property 10: Graceful Error Handling**
    - **Validates: Requirements 7.5**
  
  - [ ]* 9.4 Write unit tests for error handling
    - Test training error handling and exit codes
    - Test API error responses for various failure scenarios
    - Test file I/O error handling
    - Test logging output for different error types
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 10. Create sample iris dataset
  - [~] 10.1 Generate and save sample iris.csv file
    - Load iris dataset from sklearn
    - Convert to DataFrame with proper column names
    - Save to `data/iris.csv`
    - Verify file is non-empty and valid
    - _Requirements: 1.1_

- [ ] 11. Update requirements.txt
  - [~] 11.1 Add missing dependencies
    - Add `hypothesis` for property-based testing
    - Add `pytest` and `pytest-cov` for testing
    - Add `httpx` for FastAPI testing
    - Add `pytest-mock` for mocking
    - Ensure all existing dependencies are present
    - _Requirements: All testing requirements_

- [ ] 12. Final integration and validation
  - [~] 12.1 Run complete pipeline end-to-end
    - Run training script and verify model is saved
    - Verify MLflow artifacts are logged
    - Start API server and verify health endpoint
    - Send prediction requests and verify responses
    - Check all logs are properly formatted
    - _Requirements: All requirements_
  
  - [ ]* 12.2 Run full test suite with coverage
    - Run all unit tests
    - Run all property tests with 100 iterations
    - Generate coverage report (target: 80%+ coverage)
    - Verify all correctness properties pass
    - _Requirements: All requirements_
  
  - [ ]* 12.3 Test Docker deployment
    - Build Docker image from project root
    - Run container and verify startup
    - Test health endpoint from host
    - Test prediction endpoint from host
    - Verify logs are accessible
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [~] 13. Final checkpoint - Complete validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples, edge cases, and integration points
- The Docker build context should be the project root directory
- MLflow server should be running on http://127.0.0.1:5000 for training
- All property tests should be tagged with feature name and property number
