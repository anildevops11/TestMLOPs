# MLOps Pipeline: Visual Flow Diagrams

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MLOPS PIPELINE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   User/Dev   │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌───────────┐    ┌───────────┐   ┌───────────┐
            │   Train   │    │  Predict  │   │   View    │
            │   Model   │    │   API     │   │  MLflow   │
            └─────┬─────┘    └─────┬─────┘   └─────┬─────┘
                  │                │               │
                  │                │               │
    ┌─────────────┼────────────────┼───────────────┘
    │             │                │
    ▼             ▼                ▼
┌─────────┐  ┌─────────┐    ┌──────────┐
│  Data   │  │ Model   │    │  MLflow  │
│ Loader  │  │Registry │    │ Tracking │
└────┬────┘  └────┬────┘    └────┬─────┘
     │            │              │
     │            │              │
     ▼            ▼              ▼
┌─────────────────────────────────────┐
│         Storage Layer               │
│  • data/iris.csv                    │
│  • models/iris_model_*.pkl          │
│  • mlruns/ (experiments)            │
└─────────────────────────────────────┘
```

---

## Flow 1: Data Loading Pipeline

```
START
  │
  ▼
┌─────────────────────────┐
│ Check if iris.csv exists│
└───────┬─────────────────┘
        │
    ┌───┴───┐
    │ File  │
    │exists?│
    └───┬───┘
        │
    ┌───┴────────────────┐
    │                    │
    ▼ YES                ▼ NO
┌─────────┐         ┌──────────────┐
│ Load    │         │ Load from    │
│ CSV     │         │ sklearn      │
└────┬────┘         └──────┬───────┘
     │                     │
     ▼                     │
┌─────────────┐            │
│ Validate    │            │
│ DataFrame   │            │
└──────┬──────┘            │
       │                   │
   ┌───┴───┐               │
   │Valid? │               │
   └───┬───┘               │
       │                   │
   ┌───┴────────┐          │
   │            │          │
   ▼ YES        ▼ NO       │
┌────────┐  ┌──────────┐  │
│ Use    │  │ Fallback │  │
│ CSV    │  │ sklearn  │◄─┘
│ Data   │  │ Data     │
└───┬────┘  └────┬─────┘
    │            │
    └─────┬──────┘
          │
          ▼
    ┌──────────┐
    │ Return   │
    │ X, y     │
    └──────────┘
          │
          ▼
        END
```

### Validation Checks

```
DataFrame Validation
        │
        ├─► Check: Has 5 columns?
        │   └─► [sepal_length, sepal_width, petal_length, petal_width, species/target]
        │
        ├─► Check: All features numeric?
        │   └─► sepal_length, sepal_width, petal_length, petal_width = float/int
        │
        ├─► Check: Valid species?
        │   └─► setosa, versicolor, virginica OR 0, 1, 2
        │
        ├─► Check: No null values?
        │   └─► All cells must have values
        │
        └─► Check: Minimum rows?
            └─► At least 10 rows required
```

---

## Flow 2: Model Training Pipeline

```
START TRAINING
      │
      ▼
┌──────────────┐
│ Setup MLflow │
│ Tracking     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Load Data    │
│ (X, y)       │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Split Data       │
│ Train: 80%       │
│ Test:  20%       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Train Model      │
│ LogisticReg      │
│ max_iter=200     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Evaluate Model   │
│ • Accuracy       │
│ • Precision      │
│ • Recall         │
│ • F1-Score       │
│ • Confusion Mtx  │
└──────┬───────────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌──────────────┐         ┌──────────────┐
│ Log to       │         │ Save Model   │
│ MLflow       │         │ Locally      │
│              │         │              │
│ • Params     │         │ models/      │
│ • Metrics    │         │ iris_model_  │
│ • Artifacts  │         │ TIMESTAMP    │
│ • Model      │         │ .pkl         │
└──────┬───────┘         └──────┬───────┘
       │                        │
       └────────┬───────────────┘
                │
                ▼
          ┌──────────┐
          │ Complete │
          └──────────┘
                │
                ▼
              END
```

### MLflow Logging Details

```
MLflow Run
    │
    ├─► Parameters
    │   ├─ model_type: "LogisticRegression"
    │   ├─ max_iter: 200
    │   ├─ random_state: 42
    │   ├─ solver: "lbfgs"
    │   └─ model_path: "models/iris_model_20240220_143022.pkl"
    │
    ├─► Metrics
    │   ├─ accuracy: 0.9667
    │   ├─ precision: 0.9650
    │   ├─ recall: 0.9667
    │   └─ f1_score: 0.9655
    │
    └─► Artifacts
        ├─ model/ (sklearn model)
        └─ confusion_matrix.png
```

---

## Flow 3: Prediction Service

```
API STARTUP
    │
    ▼
┌─────────────────┐
│ Initialize      │
│ ModelPredictor  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Find Latest     │
│ Model File      │
│ (by timestamp)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Model      │
│ with joblib     │
└────────┬────────┘
         │
     ┌───┴───┐
     │Loaded?│
     └───┬───┘
         │
    ┌────┴────┐
    │         │
    ▼ YES     ▼ NO
┌─────────┐ ┌──────────┐
│ Status: │ │ Status:  │
│ healthy │ │ degraded │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           │
           ▼
    ┌──────────────┐
    │ Ready for    │
    │ Requests     │
    └──────────────┘
```

### Prediction Request Flow

```
Client Request
    │
    ▼
POST /predict
    │
    ▼
┌─────────────────┐
│ Validate Input  │
│ • 4 features?   │
│ • All numeric?  │
│ • All > 0?      │
└────────┬────────┘
         │
     ┌───┴───┐
     │Valid? │
     └───┬───┘
         │
    ┌────┴────┐
    │         │
    ▼ YES     ▼ NO
┌─────────┐ ┌──────────┐
│Continue │ │ Return   │
│         │ │ 400 Error│
└────┬────┘ └──────────┘
     │
     ▼
┌─────────────────┐
│ Check Model     │
│ Loaded?         │
└────────┬────────┘
         │
     ┌───┴───┐
     │Loaded?│
     └───┬───┘
         │
    ┌────┴────┐
    │         │
    ▼ YES     ▼ NO
┌─────────┐ ┌──────────┐
│Continue │ │ Return   │
│         │ │ 503 Error│
└────┬────┘ └──────────┘
     │
     ▼
┌─────────────────┐
│ model.predict() │
│ model.predict_  │
│ proba()         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Format Response │
│ • prediction    │
│ • index         │
│ • confidence    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return 200 OK   │
│ with JSON       │
└─────────────────┘
```

---

## Flow 4: Docker Containerization

```
BUILD PHASE
    │
    ▼
┌─────────────────┐
│ FROM python:    │
│ 3.10-slim       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ WORKDIR /app    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ COPY            │
│ requirements.txt│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RUN pip install │
│ -r requirements │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ COPY src/       │
│ COPY api/       │
│ COPY models/    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EXPOSE 8000     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HEALTHCHECK     │
│ /health         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CMD uvicorn     │
│ api.app:app     │
└────────┬────────┘
         │
         ▼
    Docker Image
    iris-api:v1
```

### Runtime Flow

```
CONTAINER START
      │
      ▼
┌──────────────┐
│ Load Python  │
│ Environment  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Import       │
│ Dependencies │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Run Startup  │
│ Event        │
│ (Load Model) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Start        │
│ Uvicorn      │
│ Server       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Listen on    │
│ Port 8000    │
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────┐      ┌──────────┐
│ Handle   │      │ Health   │
│ Requests │      │ Check    │
│          │      │ (30s)    │
└──────────┘      └──────────┘
```

---

## Flow 5: Complete End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE MLOPS WORKFLOW                   │
└─────────────────────────────────────────────────────────────┘

PHASE 1: DEVELOPMENT
    │
    ├─► Write Code
    │   ├─ data_loader.py
    │   ├─ train.py
    │   ├─ predict.py
    │   └─ app.py
    │
    ├─► Test Locally
    │   ├─ Unit tests
    │   ├─ Integration tests
    │   └─ Property tests
    │
    └─► Version Control
        └─ Git commit

PHASE 2: TRAINING
    │
    ├─► Start MLflow Server
    │   └─ mlflow ui --port 5000
    │
    ├─► Run Training
    │   └─ python src/train.py
    │
    ├─► Log to MLflow
    │   ├─ Parameters
    │   ├─ Metrics
    │   └─ Artifacts
    │
    └─► Save Model
        └─ models/iris_model_TIMESTAMP.pkl

PHASE 3: VALIDATION
    │
    ├─► Review Metrics
    │   └─ MLflow UI
    │
    ├─► Compare Models
    │   └─ Select best version
    │
    └─► Test Predictions
        └─ python -c "from src.predict import ..."

PHASE 4: DEPLOYMENT
    │
    ├─► Build Docker Image
    │   └─ docker build -t iris-api .
    │
    ├─► Test Container
    │   └─ docker run -p 8000:8000 iris-api
    │
    ├─► Push to Registry
    │   └─ docker push registry/iris-api:v1
    │
    └─► Deploy to Production
        ├─ Kubernetes
        ├─ AWS ECS
        └─ Azure Container Instances

PHASE 5: MONITORING
    │
    ├─► Health Checks
    │   └─ GET /health every 30s
    │
    ├─► Log Analysis
    │   └─ Check error rates
    │
    ├─► Performance Metrics
    │   ├─ Response time
    │   ├─ Throughput
    │   └─ Error rate
    │
    └─► Model Performance
        ├─ Prediction accuracy
        ├─ Confidence scores
        └─ Data drift detection

PHASE 6: ITERATION
    │
    ├─► Collect Feedback
    │   └─ User reports, metrics
    │
    ├─► Retrain Model
    │   └─ New data, better params
    │
    └─► Deploy New Version
        └─ Repeat from Phase 2
```

---

## Flow 6: Error Handling Paths

```
REQUEST PROCESSING
        │
        ▼
┌───────────────┐
│ Receive       │
│ Request       │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Validate      │
│ Input         │
└───────┬───────┘
        │
    ┌───┴───┐
    │Valid? │
    └───┬───┘
        │
   ┌────┴────┐
   │         │
   ▼ NO      ▼ YES
┌─────────┐ ┌──────────┐
│ Return  │ │ Continue │
│ 400     │ └────┬─────┘
│ Bad     │      │
│ Request │      ▼
└─────────┘ ┌──────────┐
            │ Check    │
            │ Model    │
            └────┬─────┘
                 │
             ┌───┴───┐
             │Loaded?│
             └───┬───┘
                 │
            ┌────┴────┐
            │         │
            ▼ NO      ▼ YES
        ┌─────────┐ ┌──────────┐
        │ Return  │ │ Predict  │
        │ 503     │ └────┬─────┘
        │ Service │      │
        │ Unavail │      ▼
        └─────────┘ ┌──────────┐
                    │ Success? │
                    └────┬─────┘
                         │
                    ┌────┴────┐
                    │         │
                    ▼ NO      ▼ YES
                ┌─────────┐ ┌──────────┐
                │ Return  │ │ Return   │
                │ 500     │ │ 200 OK   │
                │ Internal│ │ with     │
                │ Error   │ │ Result   │
                └─────────┘ └──────────┘
```

---

## Flow 7: Model Versioning Strategy

```
MODEL LIFECYCLE
      │
      ▼
┌──────────────┐
│ Train New    │
│ Model        │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Generate Timestamp   │
│ YYYYMMDD_HHMMSS      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Save as              │
│ iris_model_          │
│ 20240220_143022.pkl  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Models Directory     │
│                      │
│ ├─ iris_model_       │
│ │  20240220_120000   │
│ │  .pkl              │
│ │                    │
│ ├─ iris_model_       │
│ │  20240220_130000   │
│ │  .pkl              │
│ │                    │
│ └─ iris_model_       │
│    20240220_143022   │
│    .pkl (LATEST)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Prediction Service   │
│ Loads Latest         │
│ (by timestamp sort)  │
└──────────────────────┘
```

### Rollback Process

```
ROLLBACK SCENARIO
      │
      ▼
┌──────────────┐
│ New Model    │
│ Performs     │
│ Poorly       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Identify     │
│ Issue        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Delete Bad   │
│ Model File   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Restart API  │
│ Service      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Loads        │
│ Previous     │
│ Version      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Service      │
│ Restored     │
└──────────────┘
```

---

## Flow 8: Data Flow Through System

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA FLOW DIAGRAM                       │
└─────────────────────────────────────────────────────────────┘

INPUT DATA
    │
    ├─► iris.csv (if exists)
    │   └─► 150 rows × 5 columns
    │       ├─ sepal_length: float
    │       ├─ sepal_width: float
    │       ├─ petal_length: float
    │       ├─ petal_width: float
    │       └─ species: string
    │
    └─► sklearn.datasets.load_iris() (fallback)
        └─► 150 rows × 5 columns
            ├─ sepal_length: float
            ├─ sepal_width: float
            ├─ petal_length: float
            ├─ petal_width: float
            └─ target: int (0, 1, 2)

        ↓ VALIDATION

VALIDATED DATA
    │
    └─► Features (X): 150 × 4 DataFrame
    └─► Target (y): 150 × 1 Series

        ↓ SPLIT

TRAIN/TEST SPLIT
    │
    ├─► X_train: 120 × 4
    ├─► y_train: 120 × 1
    ├─► X_test: 30 × 4
    └─► y_test: 30 × 1

        ↓ TRAINING

TRAINED MODEL
    │
    └─► LogisticRegression object
        ├─ Coefficients
        ├─ Intercepts
        └─ Classes: [0, 1, 2]

        ↓ SAVE

PERSISTED MODEL
    │
    └─► models/iris_model_TIMESTAMP.pkl
        └─► Serialized with joblib

        ↓ LOAD

LOADED MODEL (in API)
    │
    └─► Ready for predictions

        ↓ PREDICTION

INPUT REQUEST
    │
    └─► {
        │   "sepal_length": 5.1,
        │   "sepal_width": 3.5,
        │   "petal_length": 1.4,
        │   "petal_width": 0.2
        └─► }

        ↓ PREDICT

PREDICTION OUTPUT
    │
    └─► {
        │   "prediction": "setosa",
        │   "prediction_index": 0,
        │   "confidence_scores": {
        │       "setosa": 0.98,
        │       "versicolor": 0.01,
        │       "virginica": 0.01
        │   }
        └─► }
```

---

## Flow 9: MLflow Experiment Tracking

```
EXPERIMENT LIFECYCLE
        │
        ▼
┌───────────────────┐
│ Create Experiment │
│ "iris-            │
│ classification"   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Start Run         │
│ (auto-generated   │
│ run_id)           │
└─────────┬─────────┘
          │
          ├─────────────────┬─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐      ┌─────────┐
    │   Log   │       │   Log   │      │   Log   │
    │ Params  │       │ Metrics │      │Artifacts│
    └────┬────┘       └────┬────┘      └────┬────┘
         │                 │                 │
         │                 │                 │
         ▼                 ▼                 ▼
    model_type        accuracy          model/
    max_iter          precision         confusion_
    random_state      recall            matrix.png
    solver            f1_score
    model_path
         │                 │                 │
         └─────────┬───────┴─────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ End Run      │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │ Store in     │
            │ mlruns/      │
            │ directory    │
            └──────────────┘
```

### MLflow Directory Structure

```
mlruns/
│
├─ 0/  (Default experiment)
│
└─ 1/  (iris-classification experiment)
   │
   ├─ meta.yaml
   │
   └─ abc123def456/  (Run ID)
      │
      ├─ meta.yaml
      │
      ├─ params/
      │  ├─ model_type
      │  ├─ max_iter
      │  ├─ random_state
      │  ├─ solver
      │  └─ model_path
      │
      ├─ metrics/
      │  ├─ accuracy
      │  ├─ precision
      │  ├─ recall
      │  └─ f1_score
      │
      ├─ artifacts/
      │  ├─ model/
      │  │  ├─ model.pkl
      │  │  ├─ conda.yaml
      │  │  └─ requirements.txt
      │  │
      │  └─ confusion_matrix.png
      │
      └─ tags/
         └─ mlflow.user
```

---

## Summary: Key Takeaways

### 1. Data Flow
```
CSV/sklearn → Validation → Features + Target → Train/Test Split → Model
```

### 2. Training Flow
```
Data → Train → Evaluate → Log to MLflow → Save Model → Complete
```

### 3. Prediction Flow
```
Load Model → Receive Request → Validate → Predict → Return Result
```

### 4. Deployment Flow
```
Code → Docker Image → Container → Production → Monitor
```

### 5. MLOps Cycle
```
Develop → Train → Validate → Deploy → Monitor → Iterate
```

---

**Document Version:** 1.0  
**Last Updated:** 2024-02-20  
**Companion Document:** MLOPS_LEARNING_GUIDE.md
