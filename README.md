# COMP8800-Machine Learning Intrusion Detection System using API and cloud implementation(IDS)

# Note
This system represents the current prototype implementation of the IDS system. 
Further improvements and architectural changes may be made as model evaluation 
and experimentation continue in order to determine the most effective approach.

# Prototype
It is a machine learning–based Intrusion Detection System (IDS) built using the NSL-KDD dataset.
The model is trained using a Random Forest classifier and deployed via FastAPI for real-time inference.
This prototype implements a binary classification IDS that detects whether a network connection is:

- 0 → Normal traffic
- 1 → Attack

The trained model is exposed through a REST API that supports:

- Single-record prediction
- Random test sampling
- Attack-only sampling
- Probability confidence output
- Service health status check

## Project Structure

COMP8800-project/
# API application layer
    app/                        
        api.py
# Training and preprocessing logic
    src/                        
        train.py
# Saved trained models
    model/                      
        rf_ids_pipeline.joblib
# Saved training datasets
    data/
        NSL-KDD/
            KDDTrain+.txt
            KDDTest+.txt
# Anything test related
    tests/
        make_demo_payloads.py
    requirements.txt
    .gitignore
    README.md

## Setup Instructions

### 1. Create a Virtual Environment

Windows:
    python -m venv venv
    venv\Scripts\activate

Mac/Linux:
    python -m venv venv
    source venv/bin/activate

### 2. Install Dependencies

Make sure you are in the project root folder:

    pip install -r requirements.txt


## Train the Model

To train the IDS model:

    python src/train.py

This will:

- Load NSL-KDD training data
- Perform preprocessing
- Train the Random Forest classifier
- Save the trained pipeline to:

    model/rf_ids_pipeline.joblib

## Run the API Server

Start the FastAPI server:

    uvicorn app.api:app --reload

The API will run at:

    http://127.0.0.1:8000

Interactive Swagger Documentation:

    http://127.0.0.1:8000/docs


## API Endpoints

### GET /status
Health-check endpoint verifying that the model and test dataset loaded successfully.

### POST /predict
Accepts a single flow record (features f0–f40) and returns a binary prediction with attack probability.

Example request body:

{
  "data": {
    "f0": 0,
    "f1": 181,
    ...
    "f40": 0
  }
}

### GET /predict/random
Randomly samples a row from the NSL-KDD test set and returns:

- Ground truth attack type
- Payload classification (Normal / Attack)
- Model prediction
- Attack probability

Optional parameter:

    /predict/random?attack_only=true

This filters the sampling pool to attack rows only.

## Example JSON Response

{
  "sample_index": 542,
  "true_attack_type": "neptune",
  "payload_type": "Attack",
  "prediction": 1,
  "prediction_text": "Attack",
  "attack_probability": 0.92
}

## Model Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- False Positive Rate

Minimizing false negatives is critical in IDS systems, as missed attacks represent significant security risk.

## System Architecture

NSL-KDD Dataset
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Random Forest Classifier
    ↓
Model Serialization (joblib)
    ↓
FastAPI Inference Layer
    ↓
JSON Response

flowchart TD
  A[NSL-KDD Dataset<br/>KDDTrain+.txt / KDDTest+.txt] --> B[Preprocessing & Feature Engineering<br/>src/train.py]
  B --> C[Train Random Forest Pipeline<br/>scikit-learn]
  C --> D[Save Model Artifact<br/>model/rf_ids_pipeline.joblib]

  D --> E[FastAPI Server<br/>app/api.py]
  F[Client Request<br/>POST /predict or GET /predict/random] --> E
  E --> G[Load Model + Build Input DataFrame<br/>f0..f40]
  G --> H[Model Inference<br/>predict + predict_proba]
  H --> I[JSON Response<br/>prediction + probability + metadata]

## Future Improvements

- Multi-class attack classification
- Real-time packet ingestion
- Hybrid signature + anomaly detection
- Cloud deployment (AWS EC2?Lambda?)
- Logging and monitoring
- Frontend dashboard visualization

---

## Author
Eddy Wu
February 16th, 2026