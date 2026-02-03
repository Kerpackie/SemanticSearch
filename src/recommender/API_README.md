# FFNN Recommendation API

A FastAPI-based service for serving recommendations from your trained FFNN model.

## Features

- **Score Endpoint**: Score a list of products for a given customer
- **Recommend Endpoint**: Get top-K recommendations from a candidate list
- **Automatic Feature Engineering**: Automatically computes user and item features from historical data
- **RESTful API**: Easy to integrate with any application

## Installation

1. Install required dependencies:

```bash
pip install -r requirements_api.txt
```

Or install individually:

```bash
pip install fastapi uvicorn[standard] pydantic numpy pandas scikit-learn tensorflow joblib
```

## Quick Start

### 1. Start the API Server

```bash
python api.py
```

The server will start on `http://localhost:8000`

### 2. Test the API

#### Using the Test Client

```bash
python test_api_client.py
```

#### Using curl

**Health Check:**
```bash
curl http://localhost:8000/
```

**Score Articles:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f",
    "article_ids": [108775015, 111565001, 111586001]
  }'
```

**Get Top Recommendations:**
```bash
curl -X POST "http://localhost:8000/recommend?top_k=5" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f",
    "article_ids": [108775015, 111565001, 111586001, 372860001, 464131002]
  }'
```

## API Endpoints

### `GET /`
Health check endpoint that returns model status and metadata.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/ffnn_20251016_154913/model.keras",
  "metadata": {
    "features": ["user_avg_price", "item_avg_price", "recency_days", "garment_group_name"],
    "best_val_auc": 0.8469
  }
}
```

### `POST /score`
Score a list of articles for a customer.

**Request Body:**
```json
{
  "customer_id": "customer_id_string",
  "article_ids": [108775015, 111565001, 111586001]
}
```

**Response:**
```json
{
  "customer_id": "customer_id_string",
  "scores": [
    {"article_id": 108775015, "score": 0.8523},
    {"article_id": 111565001, "score": 0.7234},
    {"article_id": 111586001, "score": 0.6341}
  ]
}
```

### `POST /recommend?top_k=N`
Get top-K recommendations from a candidate list.

**Query Parameters:**
- `top_k` (optional, default=12): Number of top recommendations to return

**Request Body:**
```json
{
  "customer_id": "customer_id_string",
  "article_ids": [108775015, 111565001, 111586001, 372860001, 464131002]
}
```

**Response:**
```json
{
  "customer_id": "customer_id_string",
  "scores": [
    {"article_id": 108775015, "score": 0.8523},
    {"article_id": 111565001, "score": 0.7234},
    {"article_id": 464131002, "score": 0.6891}
  ]
}
```

## Python Client Example

```python
import requests

# Score articles
response = requests.post(
    "http://localhost:8000/score",
    json={
        "customer_id": "your_customer_id",
        "article_ids": [108775015, 111565001, 111586001]
    }
)

result = response.json()
for item in result['scores']:
    print(f"Article {item['article_id']}: {item['score']:.4f}")
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can use these interfaces to test the API directly in your browser.

## Configuration

Edit the following constants in `api.py` to configure the service:

```python
MODEL_DIR = "models/ffnn_20251016_154913"  # Path to your trained model
TRANSACTIONS_PATH = "transactions_train.csv"  # Historical transactions
ARTICLES_PATH = "articles.csv"  # Article metadata
```

## How It Works

1. **Model Loading**: On startup, the API loads:
   - Trained Keras model
   - Scikit-learn preprocessor (for feature scaling/encoding)
   - Historical transaction data

2. **Feature Engineering**: For each prediction request:
   - Computes user features (avg price, recency)
   - Computes item features (avg price, category)
   - Merges with article metadata

3. **Prediction**: 
   - Preprocesses features using the saved preprocessor
   - Runs the neural network to generate scores
   - Returns probabilities (0-1 scale)

## Production Deployment

For production deployment, consider:

### Using Gunicorn (for better performance)

```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api.py .
COPY models/ models/
COPY transactions_train.csv .
COPY articles.csv .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t ffnn-api .
docker run -p 8000:8000 ffnn-api
```

## Performance Notes

- **Batch Predictions**: The API can handle multiple articles in a single request for efficiency
- **Caching**: User/item statistics are precomputed once at startup
- **Memory Usage**: The entire model and preprocessor are kept in memory
- **Concurrency**: Use multiple workers (gunicorn) for handling concurrent requests

## Troubleshooting

**Model not loading:**
- Check that the model path is correct
- Ensure all required files exist (model.keras, preprocessor_nn.joblib)

**Feature errors:**
- Verify that the transactions and articles CSV files are in the correct location
- Check that column names match the configuration

**Low scores:**
- The model outputs probabilities (0-1 range)
- Low scores don't necessarily mean bad - they're relative to the training data

## License

MIT

