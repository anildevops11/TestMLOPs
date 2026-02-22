# MLOps Learning Guide: Complete Step-by-Step Flow

## Table of Contents
1. [What is MLOps?](#what-is-mlops)
2. [Project Architecture Overview](#project-architecture-overview)
3. [Complete MLOps Pipeline Flow](#complete-mlops-pipeline-flow)
4. [Component Deep Dive](#component-deep-dive)
5. [Hands-On Tutorial](#hands-on-tutorial)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## What is MLOps?

**MLOps (Machine Learning Operations)** is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

### Key MLOps Principles

1. **Version Control**: Track code, data, and models
2. **Automation**: Automate training, testing, and deployment
3. **Monitoring**: Track model performance in production
4. **Reproducibility**: Ensure experiments can be recreated
5. **Collaboration**: Enable teams to work together effectively

### Why MLOps Matters

- **Without MLOps**: Models stay in notebooks, hard to deploy, no tracking
- **With MLOps**: Automated pipelines, version control, easy deployment, performance monitoring

---

## Project Architecture Overview

Our iris classification MLOps pipeline consists of 5 main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture               │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Data       │─────▶│   Training   │─────▶│   Model      │
│   Layer      │      │   Layer      │      │   Registry   │
└──────────────┘      └──────────────┘      └──────────────┘
                             │                      │
                             ▼                      ▼
                      ┌──────────────┐      ┌──────────────┐
                      │   MLflow     │      │  Prediction  │
                      │   Tracking   │      │   Service    │
                      └──────────────┘      └──────────────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │   Docker     │
                                            │   Container  │
                                            └──────────────┘
```

### Component Responsibilities

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Data Layer** | Load & validate data | `src/data_loader.py` |
| **Training Layer** | Train models | `src/train.py` |
| **Experiment Tracking** | Log metrics & artifacts | MLflow server |
| **Model Registry** | Store versioned models | `models/` directory |
| **Prediction Service** | Serve predictions | `src/predict.py`, `api/app.py` |
| **Containerization** | Package for deployment | `api/Dockerfile` |

---

## Complete MLOps Pipeline Flow

### Phase 1: Data Management

#### Step 1.1: Data Loading
```
User Request
    ↓
Load iris.csv
    ↓
File exists? ──No──▶ Load from sklearn
    ↓ Yes
Validate schema
    ↓
Valid? ──No──▶ Load from sklearn
    ↓ Yes
Return features & target
```

**What Happens:**
1. System checks if `data/iris.csv` exists and has content
2. If yes, loads CSV and validates structure
3. If no or invalid, falls back to sklearn's built-in iris dataset
4. Returns clean features (X) and target labels (y)

**Code Location:** `src/data_loader.py`

**Key Functions:**
- `load_iris_data()` - Main loading function
- `validate_iris_dataframe()` - Validates data structure

**Example:**
```python
from src.data_loader import load_iris_data

# Load data (automatically handles CSV or sklearn fallback)
X, y = load_iris_data()
print(f"Loaded {len(X)} samples with {X.shape[1]} features")
```

#### Step 1.2: Data Validation

The system validates:
- ✓ Exactly 5 columns (4 features + 1 target)
- ✓ Column names: sepal_length, sepal_width, petal_length, petal_width, species/target
- ✓ All features are numeric
- ✓ Target contains only valid species (setosa, versicolor, virginica)
- ✓ No missing values
- ✓ At least 10 rows

**Why This Matters:**
Bad data = bad models. Validation catches issues early before wasting time training.

---

### Phase 2: Model Training with Experiment Tracking

#### Step 2.1: Training Pipeline Flow

```
Start Training
    ↓
Setup MLflow ──────────────▶ Connect to tracking server
    ↓                        (http://127.0.0.1:5000)
Load Data
    ↓
Split Train/Test (80/20)
    ↓
Train Model ───────────────▶ Logistic Regression
    ↓
Evaluate Model ────────────▶ Calculate metrics
    ↓                        (accuracy, precision, recall, F1)
Log to MLflow ─────────────▶ Save params, metrics, artifacts
    ↓
Save Model Locally ────────▶ models/iris_model_TIMESTAMP.pkl
    ↓
Complete
```

**Code Location:** `src/train.py`

#### Step 2.2: MLflow Experiment Tracking

**What is MLflow?**
MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.

**What Gets Tracked:**

1. **Parameters** (hyperparameters):
   ```python
   mlflow.log_param("model_type", "LogisticRegression")
   mlflow.log_param("max_iter", 200)
   mlflow.log_param("random_state", 42)
   ```

2. **Metrics** (performance):
   ```python
   mlflow.log_metric("accuracy", 0.95)
   mlflow.log_metric("precision", 0.94)
   mlflow.log_metric("recall", 0.95)
   mlflow.log_metric("f1_score", 0.94)
   ```

3. **Artifacts** (files):
   ```python
   mlflow.sklearn.log_model(model, "model")
   mlflow.log_artifact("confusion_matrix.png")
   ```

4. **Model Path**:
   ```python
   mlflow.log_param("model_path", "models/iris_model_20240220_143022.pkl")
   ```

**Why MLflow?**
- Compare different model versions
- Track what hyperparameters produced best results
- Reproduce experiments exactly
- Share results with team

#### Step 2.3: Model Versioning

Models are saved with timestamp-based naming:
```
models/
├── iris_model_20240220_120000.pkl  (older)
├── iris_model_20240220_130000.pkl
└── iris_model_20240220_143022.pkl  (latest)
```

**Format:** `iris_model_YYYYMMDD_HHMMSS.pkl`

**Benefits:**
- Easy to identify latest model
- Can rollback to previous versions
- Track model evolution over time

---

### Phase 3: Model Serving (Prediction Service)

#### Step 3.1: Prediction Flow

```
API Startup
    ↓
Load Latest Model ─────────▶ Find most recent timestamp
    ↓                        models/iris_model_*.pkl
Model Loaded?
    ↓ Yes                    ↓ No
Set status: healthy          Set status: degraded
    ↓
Ready for Predictions
    ↓
Receive Request ───────────▶ POST /predict
    ↓                        {sepal_length, sepal_width,
Validate Input               petal_length, petal_width}
    ↓
Make Prediction ───────────▶ model.predict()
    ↓                        model.predict_proba()
Return Response ───────────▶ {prediction, confidence_scores}
```

**Code Location:** `src/predict.py`, `api/app.py`

#### Step 3.2: ModelPredictor Class

```python
class ModelPredictor:
    def __init__(self):
        self.model = None
        self.model_path = None
    
    def load_latest_model(self):
        """Finds and loads the most recent model by timestamp"""
        # Find all model files
        model_files = sorted(MODEL_DIR.glob("iris_model_*.pkl"))
        # Load the latest one
        self.model = joblib.load(model_files[-1])
    
    def predict(self, features):
        """Generate prediction with confidence scores"""
        prediction = self.model.predict([features])
        probabilities = self.model.predict_proba([features])
        return {
            "prediction": species_name,
            "confidence_scores": {species: prob}
        }
```

**Key Features:**
- Automatic latest model selection
- Confidence scores for all classes
- Error handling for missing models

---

### Phase 4: REST API with FastAPI

#### Step 4.1: API Endpoints

**1. Health Check Endpoint**
```
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/iris_model_20240220_143022.pkl"
}
```

**Purpose:** Check if service is running and model is loaded

**2. Prediction Endpoint**
```
POST /predict

Request:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Response:
{
  "prediction": "setosa",
  "prediction_index": 0,
  "confidence_scores": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  }
}
```

**Purpose:** Get species prediction for iris measurements

#### Step 4.2: Request Validation

FastAPI uses Pydantic models for automatic validation:

```python
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")
```

**Validation Rules:**
- All fields required
- Must be float type
- Must be greater than 0

**Invalid Request Example:**
```json
{
  "sepal_length": -1.0,  // ❌ Negative value
  "sepal_width": "abc"   // ❌ Not a number
}
```

Returns: `400 Bad Request` with validation details

#### Step 4.3: Error Handling

| Error Type | HTTP Status | Response |
|------------|-------------|----------|
| Invalid input | 400 | Validation error details |
| Model not loaded | 503 | "Model not available" |
| Prediction error | 500 | "Internal server error" |
| Success | 200 | Prediction result |

---

### Phase 5: Containerization with Docker

#### Step 5.1: Docker Container Flow

```
Build Docker Image
    ↓
FROM python:3.10-slim ─────▶ Base image
    ↓
COPY requirements.txt ─────▶ Copy dependencies
    ↓
RUN pip install ───────────▶ Install packages
    ↓
COPY src/, api/, models/ ──▶ Copy application code
    ↓
EXPOSE 8000 ───────────────▶ Open port
    ↓
CMD uvicorn ───────────────▶ Start FastAPI server
    ↓
Container Running
    ↓
Health Check Every 30s ────▶ GET /health
```

**Code Location:** `api/Dockerfile`

#### Step 5.2: Dockerfile Explained

```dockerfile
# Start with Python 3.10 slim image (smaller size)
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose port 8000 for API
EXPOSE 8000

# Health check: ping /health every 30 seconds
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 5.3: Docker Commands

**Build Image:**
```bash
docker build -f api/Dockerfile -t iris-api .
```

**Run Container:**
```bash
docker run -p 8000:8000 iris-api
```

**Test Container:**
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## Component Deep Dive

### 1. Data Loader (`src/data_loader.py`)

**Purpose:** Ensure high-quality data enters the pipeline

**Key Concepts:**

**Fallback Pattern:**
```python
try:
    # Try primary data source (CSV)
    data = load_from_csv()
except:
    # Fallback to secondary source (sklearn)
    data = load_from_sklearn()
```

**Why?** Ensures pipeline never fails due to missing data

**Validation Pattern:**
```python
def validate_data(df):
    if not meets_requirements(df):
        raise ValueError("Descriptive error message")
    return True
```

**Why?** Fail fast with clear error messages

### 2. Training Service (`src/train.py`)

**Purpose:** Train models with full experiment tracking

**Key Concepts:**

**MLflow Context Manager:**
```python
with mlflow.start_run():
    # Everything here gets logged to this run
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", score)
    mlflow.log_artifact("file.png")
```

**Why?** Groups all related logs together

**Separation of Concerns:**
```python
def train_model():      # Only training logic
def evaluate_model():   # Only evaluation logic
def save_model():       # Only saving logic
def main():             # Orchestrates everything
```

**Why?** Easier to test, modify, and understand

### 3. Prediction Service (`src/predict.py`)

**Purpose:** Load models and generate predictions

**Key Concepts:**

**Lazy Loading:**
```python
class ModelPredictor:
    def __init__(self):
        self.model = None  # Don't load yet
    
    def load_latest_model(self):
        # Load only when needed
        self.model = joblib.load(path)
```

**Why?** Faster startup, load only when necessary

**Confidence Scores:**
```python
probabilities = model.predict_proba(features)
# Returns: [0.98, 0.01, 0.01] for each class
```

**Why?** Know how confident the model is

### 4. FastAPI Application (`api/app.py`)

**Purpose:** Expose ML model as REST API

**Key Concepts:**

**Startup Events:**
```python
@app.on_event("startup")
async def startup_event():
    # Load model when server starts
    predictor.load_latest_model()
```

**Why?** Load heavy resources once, not per request

**Dependency Injection:**
```python
# Predictor created once at module level
predictor = ModelPredictor()

# All endpoints use same instance
@app.post("/predict")
async def predict():
    return predictor.predict(data)
```

**Why?** Share resources across requests

**Graceful Degradation:**
```python
if not model_loaded:
    return {"status": "degraded"}  # Still respond
```

**Why?** Service stays up even if model fails

---

## Hands-On Tutorial

### Tutorial 1: Running the Complete Pipeline

**Step 1: Start MLflow Server**
```bash
cd mlflow
./mlflow_server.sh
# Or: mlflow ui --host 0.0.0.0 --port 5000
```

**What This Does:**
- Starts MLflow tracking server on http://localhost:5000
- Creates `mlruns/` directory to store experiments
- Provides web UI to view experiments

**Step 2: Train a Model**
```bash
python src/train.py
```

**What Happens:**
1. Loads iris data (CSV or sklearn fallback)
2. Splits into train/test sets
3. Trains logistic regression model
4. Evaluates performance
5. Logs everything to MLflow
6. Saves model to `models/iris_model_TIMESTAMP.pkl`

**Expected Output:**
```
2024-02-20 14:30:22 - __main__ - INFO - Starting MLOps training pipeline
2024-02-20 14:30:22 - __main__ - INFO - MLflow configured with tracking URI: http://127.0.0.1:5000
2024-02-20 14:30:22 - data_loader - INFO - Loading iris dataset from sklearn
2024-02-20 14:30:22 - __main__ - INFO - Dataset loaded: 150 samples, 4 features
2024-02-20 14:30:22 - __main__ - INFO - Training logistic regression model...
2024-02-20 14:30:22 - __main__ - INFO - Accuracy: 0.9667
2024-02-20 14:30:22 - __main__ - INFO - Model saved to models/iris_model_20240220_143022.pkl
```

**Step 3: View Experiments in MLflow**
1. Open browser: http://localhost:5000
2. Click on "iris-classification" experiment
3. See all runs with metrics and parameters
4. Compare different runs
5. Download artifacts (model, confusion matrix)

**Step 4: Start Prediction API**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**What This Does:**
- Loads latest model from `models/` directory
- Starts FastAPI server on http://localhost:8000
- Exposes /health and /predict endpoints

**Step 5: Test the API**

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/iris_model_20240220_143022.pkl"
}
```

**Make Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": "setosa",
  "prediction_index": 0,
  "confidence_scores": {
    "setosa": 0.9834,
    "versicolor": 0.0123,
    "virginica": 0.0043
  }
}
```

**Step 6: Interactive API Documentation**
Open browser: http://localhost:8000/docs

FastAPI automatically generates interactive API documentation where you can:
- See all endpoints
- Try requests directly in browser
- View request/response schemas

---

### Tutorial 2: Docker Deployment

**Step 1: Build Docker Image**
```bash
docker build -f api/Dockerfile -t iris-api:v1 .
```

**What This Does:**
- Reads Dockerfile instructions
- Creates isolated environment with Python 3.10
- Installs all dependencies
- Copies application code
- Creates runnable container image

**Step 2: Run Container**
```bash
docker run -d -p 8000:8000 --name iris-service iris-api:v1
```

**Flags Explained:**
- `-d`: Run in background (detached mode)
- `-p 8000:8000`: Map container port 8000 to host port 8000
- `--name iris-service`: Give container a friendly name
- `iris-api:v1`: Image name and tag

**Step 3: Check Container Status**
```bash
docker ps
```

**Output:**
```
CONTAINER ID   IMAGE         STATUS                    PORTS
abc123def456   iris-api:v1   Up 2 minutes (healthy)    0.0.0.0:8000->8000/tcp
```

**Step 4: View Container Logs**
```bash
docker logs iris-service
```

**Step 5: Test Containerized API**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Step 6: Stop and Remove Container**
```bash
docker stop iris-service
docker rm iris-service
```

---

### Tutorial 3: Model Versioning and Rollback

**Scenario:** You trained a new model but it performs worse. Roll back to previous version.

**Step 1: List All Models**
```bash
ls -lt models/
```

**Output:**
```
iris_model_20240220_150000.pkl  (latest - bad model)
iris_model_20240220_140000.pkl  (previous - good model)
iris_model_20240220_130000.pkl  (older)
```

**Step 2: Delete Bad Model**
```bash
rm models/iris_model_20240220_150000.pkl
```

**Step 3: Restart API**
```bash
# API will now load iris_model_20240220_140000.pkl (latest remaining)
uvicorn api.app:app --reload
```

**Step 4: Verify Rollback**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/iris_model_20240220_140000.pkl"
}
```

---

## Best Practices

### 1. Data Management

**✓ DO:**
- Validate data before training
- Use fallback data sources
- Log data source used
- Version your datasets

**✗ DON'T:**
- Train on unvalidated data
- Ignore missing values
- Mix different data formats
- Lose track of data versions

### 2. Experiment Tracking

**✓ DO:**
- Log all hyperparameters
- Track all metrics
- Save model artifacts
- Use descriptive experiment names

**✗ DON'T:**
- Skip logging experiments
- Forget to log hyperparameters
- Overwrite previous experiments
- Use generic names like "test1", "test2"

### 3. Model Versioning

**✓ DO:**
- Use timestamp-based naming
- Keep multiple versions
- Document model changes
- Test before deploying

**✗ DON'T:**
- Overwrite existing models
- Use generic names like "model.pkl"
- Delete old versions immediately
- Deploy untested models

### 4. API Design

**✓ DO:**
- Validate all inputs
- Return confidence scores
- Implement health checks
- Use proper HTTP status codes

**✗ DON'T:**
- Trust user input
- Return only predictions
- Skip health monitoring
- Use 200 for all responses

### 5. Error Handling

**✓ DO:**
- Catch specific exceptions
- Log errors with context
- Return user-friendly messages
- Implement graceful degradation

**✗ DON'T:**
- Use bare except clauses
- Expose stack traces to users
- Crash on errors
- Fail silently

### 6. Containerization

**✓ DO:**
- Use specific base images
- Minimize image size
- Implement health checks
- Use multi-stage builds (advanced)

**✗ DON'T:**
- Use latest tags
- Include unnecessary files
- Skip health checks
- Hardcode configurations

---

## Troubleshooting

### Issue 1: MLflow Connection Failed

**Error:**
```
ConnectionError: Cannot connect to MLflow server at http://127.0.0.1:5000
```

**Solution:**
```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Or check if already running
netstat -an | findstr 5000  # Windows
lsof -i :5000               # Linux/Mac
```

### Issue 2: Model Not Found

**Error:**
```
No model files found in models/ directory
```

**Solution:**
```bash
# Train a model first
python src/train.py

# Verify model exists
ls models/
```

### Issue 3: API Returns 503

**Error:**
```json
{
  "status": "degraded",
  "model_loaded": false
}
```

**Solution:**
1. Check if model exists: `ls models/`
2. Check API logs for errors
3. Verify model file is not corrupted
4. Retrain model if necessary

### Issue 4: Docker Build Fails

**Error:**
```
COPY failed: file not found
```

**Solution:**
```bash
# Build from project root, not api/ directory
cd /path/to/project
docker build -f api/Dockerfile -t iris-api .
```

### Issue 5: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'data_loader'
```

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or run from project root
cd /path/to/project
python src/train.py
```

---

## Summary: Complete MLOps Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    End-to-End MLOps Flow                     │
└─────────────────────────────────────────────────────────────┘

1. DATA MANAGEMENT
   ├─ Load iris.csv (or sklearn fallback)
   ├─ Validate schema and content
   └─ Return clean features and target

2. MODEL TRAINING
   ├─ Setup MLflow tracking
   ├─ Split train/test data
   ├─ Train logistic regression
   ├─ Evaluate with multiple metrics
   ├─ Log params, metrics, artifacts to MLflow
   └─ Save versioned model (timestamp-based)

3. EXPERIMENT TRACKING (MLflow)
   ├─ Track all hyperparameters
   ├─ Log performance metrics
   ├─ Store model artifacts
   ├─ Save confusion matrix
   └─ Enable experiment comparison

4. MODEL SERVING
   ├─ Load latest model by timestamp
   ├─ Expose REST API endpoints
   ├─ Validate incoming requests
   ├─ Generate predictions with confidence
   └─ Handle errors gracefully

5. CONTAINERIZATION
   ├─ Package application in Docker
   ├─ Include all dependencies
   ├─ Implement health checks
   ├─ Deploy consistently anywhere
   └─ Scale horizontally

6. MONITORING & MAINTENANCE
   ├─ Health check endpoint
   ├─ Structured logging
   ├─ Error tracking
   ├─ Model versioning
   └─ Easy rollback capability
```

---

## Next Steps for Learning

### Beginner Level
1. ✓ Understand the complete flow (this document)
2. Run the pipeline locally
3. Make predictions via API
4. View experiments in MLflow UI
5. Build and run Docker container

### Intermediate Level
1. Modify hyperparameters and compare results
2. Add new features to the model
3. Implement A/B testing (serve multiple models)
4. Add monitoring and alerting
5. Set up CI/CD pipeline

### Advanced Level
1. Implement model retraining automation
2. Add data drift detection
3. Set up model performance monitoring
4. Implement canary deployments
5. Scale with Kubernetes

---

## Additional Resources

### Official Documentation
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://docs.docker.com/
- **Scikit-learn**: https://scikit-learn.org/

### Recommended Reading
- "Designing Machine Learning Systems" by Chip Huyen
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- "Machine Learning Engineering" by Andriy Burkov

### Video Tutorials
- MLflow Tutorial: https://www.youtube.com/watch?v=859OxXrt_TI
- FastAPI Tutorial: https://www.youtube.com/watch?v=0sOvCWFmrtA
- Docker for ML: https://www.youtube.com/watch?v=0qG_0CPQhpg

---

## Glossary

| Term | Definition |
|------|------------|
| **MLOps** | Machine Learning Operations - practices for deploying and maintaining ML systems |
| **MLflow** | Open-source platform for managing ML lifecycle |
| **Experiment Tracking** | Recording parameters, metrics, and artifacts from ML experiments |
| **Model Registry** | Centralized store for managing model versions |
| **REST API** | Web service that uses HTTP requests to access and manipulate data |
| **FastAPI** | Modern Python web framework for building APIs |
| **Docker** | Platform for packaging applications in containers |
| **Container** | Lightweight, standalone package with everything needed to run software |
| **Health Check** | Endpoint that reports service status |
| **Graceful Degradation** | System continues operating even when components fail |
| **Versioning** | Tracking different versions of code, data, or models |
| **Hyperparameters** | Configuration values that control model training |
| **Artifacts** | Files produced during ML experiments (models, plots, etc.) |
| **Inference** | Using a trained model to make predictions |
| **Endpoint** | URL where API can be accessed |

---

**Document Version:** 1.0  
**Last Updated:** 2024-02-20  
**Author:** MLOps Pipeline Team
