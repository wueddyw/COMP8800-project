# COMP8800-Machine Learning Intrusion Detection System using API and cloud implementation(IDS)

# Overview
This project implements a machine learning–based Intrusion Detection System (IDS) using the NSL-KDD dataset.
The system is designed to detect malicious network activity using a hybrid detection approach, combining:

- Random Forest (RF) for high-confidence detection (signature-like behavior)
- Deep Neural Network (DNN) for uncertain or complex patterns (anomaly detection)

The system is deployed via FastAPI to support real-time inference through a REST API.

# Note
This system represents a work in progress with 1/5 milesontes implemented of the IDS system. 
Further improvements and architectural changes may be made as model evaluation 
and experimentation continue in order to determine the most effective approach.

# Detection Approach

## Random Forest (Stage 1)
- Handles high-confidence predictions
- Efficient and interpretable
- Acts as a signature-like detection layer

## Deep Neural Network (Stage 2)
- Activated for uncertain RF predictions
- Learns complex, non-linear attack patterns
- Represents anomaly detection component

## Hybrid Logic
- If RF confidence is high → use RF result
- If RF confidence is low → route to DNN


# Prototype (Done)
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

## Project Structure (Prototype)

COMP8800-project/
API application layer
    app/                        
        api.py
Training and preprocessing logic
    src/                        
        train.py
Saved trained models
    model/                      
        rf_ids_pipeline.joblib
Saved training datasets
    data/
        NSL-KDD/
            KDDTrain+.txt
            KDDTest+.txt
Anything test related
    tests/
        make_demo_payloads.py
    requirements.txt
    .gitignore
    README.md

# Milestone 1 (Current)
Milestone 1 focuses on improving the baseline model and introducing hybrid architecture components.

## Progress 
- Built baseline Random Forest IDS prototype
- Applied hyperparameter tuning to improve model performance
- Implemented threshold tuning to optimize attack detection
- Designed and integrated an initial Deep Neural Network (DNN)
- Developed hybrid routing logic between RF and DNN
- Evaluated hybrid system performance and identified improvement areas

# Project Structure (Milestone 1)
COMP8800-project/
app/                    
    api.py
src/                    
    train_rf_ids.py
    train_hybrid_ids.py
model/                
    rf_ids_pipeline.joblib
    rf_ids_pipeline_tuned.joblib
    hybrid_dnn_model.keras
data/
    NSL-KDD/
        KDDTrain+.txt
        KDDTest+.txt
tests/
    make_demo_payloads.py
out/                    
docs/                      
requirements.txt
README.md

# Model progression

1. Prototype Random Forest
- Accuracy: 77.6%
- Attack Recall: 0.63

2. Tuned Random Forest
- Accuracy: 80.3%
- Attack Recall: 0.68

3. Tuned RF + Threshold Optimization
- Accuracy: 83.3%
- Attack Recall: 0.74

4. Initial Hybrid RF + DNN
- Successfully integrated DNN into pipeline
- Hybrid routing implemented (confidence-based)
- Initial results indicate further tuning required

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

Train Random Forest (Baseline / Tuned):

    python src/train_rf_ids.py

Train Hybrid Model (RF + DNN):

    python src/train_hybrid_ids.py  

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
Random Forest (Stage 1)
    ↓
Confidence Check
    ↓           ↓
    High        Low
    ↓           ↓
    Output      DNN (Stage 2)
                ↓
                Final Prediction
                ↓
                FastAPI response

# Limitations (Milestone 1)

- DNN trained on limited uncertain samples
- No feature scaling applied for neural network
- Hybrid model does not yet outperform optimized RF


## Future Improvements

- Proper feature scaling for DNN (StandardScaler)
- Train DNN on larger dataset
- Multi-class attack classification
- Real-time packet capture integration
- AWS deployment (EC2 / Lambda)
- Monitoring and logging pipeline
- Dashboard visualization

## Author
Eddy Wu
February 16th, 2026
