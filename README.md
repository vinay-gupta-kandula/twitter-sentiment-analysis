# Twitter Sentiment Analysis

A complete sentiment analysis system built using BERT fine-tuning, FastAPI, Streamlit, and Docker. The project provides a machine learning pipeline for preprocessing text, fine-tuning a pre-trained BERT model, evaluating the model, serving predictions through a REST API, and providing an interactive web interface.

> **Note:** The project is named Twitter Sentiment Analysis, but the training dataset used for this implementation is the IMDb dataset from Hugging Face. The model therefore performs binary positive/negative sentiment classification on text and is not specifically trained on Twitter data.

---

## Features

- Text preprocessing using the Hugging Face IMDb dataset
- BERT fine-tuning for binary sentiment classification
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- FastAPI REST API
- Interactive Streamlit web interface
- Batch prediction from CSV files
- Dockerized API and UI
- Docker Compose orchestration
- API and UI health checks
- Automated API tests using pytest
- Environment-based configuration
- Model artifacts saved separately from source code
- Training performed separately from deployment

---

## Architecture

```text
                    +----------------------+
                    |   Input Text / CSV   |
                    +----------+-----------+
                               |
                +--------------v--------------+
                |       BERT Model            |
                |  Fine-tuned for Sentiment   |
                +--------------+--------------+
                               |
              +----------------+----------------+
              |                                 |
     +--------v---------+              +--------v---------+
     |    FastAPI       |              | Batch Prediction |
     |    REST API      |              |     Script       |
     +--------+---------+              +------------------+
              |
     +--------v---------+
     |    Streamlit     |
     |       UI         |
     +------------------+

Docker Compose
├── API Container
└── UI Container
````

---

## Technology Stack

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* BERT (`bert-base-uncased`)
* Scikit-learn
* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit
* Requests

### DevOps

* Docker
* Docker Compose
* Environment variables

### Testing

* Pytest
* FastAPI TestClient

---

## Project Structure

```text
twitter-sentiment-analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   └── unseen/
│       ├── test_input.csv
│       └── predictions.csv
│
├── model_output/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── results/
│   ├── metrics.json
│   └── run_summary.json
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── batch_predict.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── ui.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
├── requirements.api.txt
├── requirements.ui.txt
└── README.md
```

---

# 1. Dataset and Preprocessing

The project uses the IMDb sentiment dataset provided through Hugging Face Datasets.

The preprocessing script:

```text
scripts/preprocess.py
```

performs the following operations:

1. Downloads the IMDb dataset.
2. Loads the training and testing splits.
3. Removes URLs.
4. Removes HTML tags.
5. Removes user mentions.
6. Normalizes whitespace.
7. Removes empty text entries.
8. Saves the processed datasets as CSV files.

Generated files:

```text
data/processed/train.csv
data/processed/test.csv
```

Each CSV contains:

```text
text,label
```

The processed dataset contains:

* 25,000 training samples
* 25,000 testing samples
* Binary labels: `0` and `1`

To run preprocessing:

```powershell
python scripts/preprocess.py
```

---

# 2. BERT Fine-Tuning

The model used for sentiment classification is:

```text
bert-base-uncased
```

Training is implemented in:

```text
scripts/train.py
```

The training pipeline:

1. Loads the processed CSV files.
2. Loads the pre-trained BERT tokenizer.
3. Tokenizes the text.
4. Uses a maximum sequence length of 128.
5. Loads BERT for sequence classification.
6. Fine-tunes the model on the sentiment dataset.
7. Evaluates the model.
8. Saves the model and tokenizer.
9. Saves evaluation metrics.
10. Saves the training configuration and final metrics.

### Training Configuration

```text
Model: bert-base-uncased
Learning rate: 2e-5
Batch size: 8
Epochs: 1
Maximum sequence length: 128
Weight decay: 0.01
```

---

# 3. Training Environment

The project was initially set up and prepared locally on Windows. This included creating a Python virtual environment, installing the dependencies, running preprocessing, and generating the processed training and testing files. Each split contains 25,000 samples.

The first local training attempt used CPU. Since fine-tuning `bert-base-uncased` on 25,000 training samples is time-consuming without hardware acceleration, the final fine-tuning run was performed in Google Colab using a Tesla T4 GPU.

The training used mixed precision (FP16).

The Colab run completed in approximately five minutes. Its final evaluation results were:

| Metric | Score |
| --- | ---: |
| Accuracy | 89.02% |
| Precision | 88.00% |
| Recall | 90.37% |
| F1 Score | 89.17% |

After training, the model artifacts, tokenizer files, and evaluation results were downloaded from Colab and placed into:

```text
model_output/
results/
```

The local `scripts/train.py` remains the reproducible training script, while Google Colab served as the accelerated training environment. All remaining development and verification was completed locally, including the FastAPI backend, Streamlit UI, batch prediction script, Dockerfiles, Docker Compose configuration, and pytest tests. The complete application was then verified with Docker Compose, with the API and UI running as separate communicating containers.

To reproduce training in another environment:

```powershell
python scripts/train.py
```

For GPU training, ensure that the environment has a compatible CUDA-enabled PyTorch installation.

---

# 4. Model Evaluation

The final evaluation metrics from the trained BERT model are:

| Metric    |    Score |
| --------- | -------: |
| Accuracy  | 0.890240 |
| Precision | 0.880025 |
| Recall    | 0.903680 |
| F1 Score  | 0.891696 |

The metrics are stored in:

```text
results/metrics.json
```

Example:

```json
{
  "accuracy": 0.89024,
  "precision": 0.880025,
  "recall": 0.90368,
  "f1_score": 0.891696
}
```

Training configuration and final metrics are stored in:

```text
results/run_summary.json
```

---

# 5. Model Artifacts

The trained model is stored in:

```text
model_output/
```

Current artifacts include:

```text
model_output/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The model artifact is approximately 438 MB.

Because the model is large, `model_output/` is excluded from Git using `.gitignore`.

If the repository is cloned without the model, the model must be generated using the training process and placed in:

```text
model_output/
```

before starting the API container.

---

# 6. FastAPI Backend

The API implementation is:

```text
src/api.py
```

The API loads the trained model once when the application starts.

## Start API Locally

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Health Endpoint

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok"
}
```

---

## Prediction Endpoint

### Request

```http
POST /predict
```

Request body:

```json
{
  "text": "I absolutely love this product!"
}
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.996
}
```

The API supports:

* Non-empty text validation
* Maximum text length validation
* Positive/negative classification
* Confidence score between 0 and 1

---

# 7. Streamlit Web Interface

The Streamlit UI is implemented in:

```text
src/ui.py
```

The UI sends the entered text to the FastAPI backend and displays:

* Predicted sentiment
* Confidence score

Run locally:

```powershell
streamlit run src/ui.py
```

The application is available at:

```text
http://localhost:8501
```

---

# 8. Batch Prediction

Batch prediction is implemented in:

```text
scripts/batch_predict.py
```

The input CSV must contain a:

```text
text
```

column.

Example:

```csv
text
"I absolutely love this!"
"This product is terrible."
"The service was okay."
```

Run:

```powershell
python scripts/batch_predict.py --input data/unseen/test_input.csv --output data/unseen/predictions.csv --model model_output
```

Output:

```csv
text,predicted_sentiment,confidence
I absolutely love this!,positive,0.9959
This product is terrible.,negative,0.9951
The service was okay.,positive,0.8344
```

---

# 9. Docker

The project contains separate Dockerfiles for the API and UI:

```text
Dockerfile.api
Dockerfile.ui
```

Docker Compose is defined in:

```text
docker-compose.yml
```

The Compose configuration contains two services:

```text
api
ui
```

The UI depends on the API health check before starting.

---

## Build and Start

Make sure Docker Desktop is running.

From the project root:

```powershell
docker compose up --build -d
```

Check the containers:

```powershell
docker compose ps
```

Both services should eventually show:

```text
healthy
```

---

## Docker Services

### API

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

### UI

```text
http://localhost:8501
```

---

## Stop Containers

```powershell
docker compose down
```

---

## View Logs

API:

```powershell
docker compose logs api
```

UI:

```powershell
docker compose logs ui
```

Both services can also be viewed together:

```powershell
docker compose logs
```

---

# 10. Environment Variables

Example configuration is provided in:

```text
.env.example
```

```env
API_PORT=8000
UI_PORT=8501
MODEL_PATH=model_output
API_URL=http://api:8000
```

These values provide configuration for the API model path and UI-to-API communication.

---

# 11. Requirements

The project separates deployment dependencies from training dependencies.

### Training / preprocessing

```text
requirements.txt
```

### API

```text
requirements.api.txt
```

### UI

```text
requirements.ui.txt
```

This prevents the UI container from installing unnecessary machine learning dependencies.

---

# 12. Local Development Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run preprocessing:

```powershell
python scripts/preprocess.py
```

Train the model:

```powershell
python scripts/train.py
```

Start the API:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

In another terminal, start the UI:

```powershell
streamlit run src/ui.py
```

---

# 13. Testing

API tests are located in:

```text
tests/test_api.py
```

Install pytest if required:

```powershell
pip install pytest
```

Run:

```powershell
python -m pytest
```

The test suite verifies:

* `/health`
* `/predict`
* Empty text validation

Current test result:

```text
3 passed
```

---

# 14. End-to-End Flow

The complete application flow is:

```text
User
 |
 v
Streamlit UI
 |
 | POST /predict
 v
FastAPI
 |
 v
BERT Tokenizer
 |
 v
Fine-tuned BERT Model
 |
 v
Sentiment + Confidence
 |
 v
FastAPI
 |
 v
Streamlit UI
```

For batch prediction:

```text
CSV
 |
 v
batch_predict.py
 |
 v
Tokenizer
 |
 v
BERT Model
 |
 v
Prediction CSV
```

---

# 15. Deployment Workflow

The recommended workflow is:

```text
1. Prepare dataset
        |
        v
2. Preprocess data
        |
        v
3. Fine-tune BERT
        |
        v
4. Evaluate model
        |
        v
5. Save model artifacts
        |
        v
6. Build Docker images
        |
        v
7. Start API and UI
        |
        v
8. Run health checks
        |
        v
9. Test predictions
```

Training is intentionally separate from the API deployment process. The API Docker image uses the already-trained model artifacts rather than training the model during container startup.

---

# 16. Important Notes

* The model is trained for binary sentiment classification.
* Label `0` is mapped to `negative`.
* Label `1` is mapped to `positive`.
* The model is trained using IMDb text rather than a Twitter-specific dataset.
* The model is therefore suitable for demonstrating the sentiment-analysis pipeline, but its training data should be considered when interpreting predictions on Twitter-specific language.
* The trained model is excluded from Git because of its large file size.
* Docker deployment requires the trained model artifacts to exist in `model_output/`.

---

# 17. Troubleshooting

### Docker containers are not starting

Check:

```powershell
docker compose ps
```

Then inspect logs:

```powershell
docker compose logs api
docker compose logs ui
```

### API cannot find the model

Verify:

```powershell
Get-ChildItem .\model_output
```

The directory should contain the trained model and tokenizer files.

### UI cannot connect to API

When running through Docker Compose, the UI uses:

```text
http://api:8000
```

When running the UI directly on the host, use:

```text
http://localhost:8000
```

### Tests cannot import the project

Make sure the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python -m pytest
```

---

# 18. Project Status

The project currently includes:

* BERT fine-tuning pipeline
* Processed training and testing datasets
* Trained BERT model
* Evaluation metrics
* FastAPI backend
* Streamlit frontend
* Batch prediction
* Docker API container
* Docker UI container
* Docker Compose orchestration
* Health checks
* Automated API tests
* Environment configuration
* Project documentation

The complete application can be started with:

```powershell
docker compose up --build -d
```

