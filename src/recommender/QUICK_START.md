# FFNN Recommendation API - Quick Start Guide

## 🎯 Overview

Your recommendation API is ready! It wraps your trained FFNN model in a REST API that allows you to:
- Send a customer ID and a list of article IDs
- Receive back scores (probabilities) for each article
- Get top-K recommendations from a candidate list

## 📁 Created Files

```
api.py                      # Main FastAPI server
test_api_client.py         # Python test client
api_demo.ipynb             # Jupyter notebook demo
validate_api_setup.py      # Setup validation script
start_api.sh               # Startup script
requirements_api.txt       # Python dependencies
API_README.md              # Detailed documentation
```

## 🚀 Quick Start (3 Steps)

### Step 1: Start the API Server

```bash
python api.py
```

Or use the startup script:
```bash
./start_api.sh
```

The server will start on `http://localhost:8000`

### Step 2: Test the API

Open a new terminal and run:
```bash
python test_api_client.py
```

### Step 3: Use the API

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/score",
    json={
        "customer_id": "your_customer_id_here",
        "article_ids": [108775015, 111565001, 111586001]
    }
)

result = response.json()
for item in result['scores']:
    print(f"Article {item['article_id']}: {item['score']:.4f}")
```

## 📊 API Endpoints

### `POST /score`
Score all articles for a customer.

**Request:**
```json
{
  "customer_id": "customer_123",
  "article_ids": [108775015, 111565001, 111586001]
}
```

**Response:**
```json
{
  "customer_id": "customer_123",
  "scores": [
    {"article_id": 108775015, "score": 0.8523},
    {"article_id": 111565001, "score": 0.7234},
    {"article_id": 111586001, "score": 0.6341}
  ]
}
```

### `POST /recommend?top_k=N`
Get top-K recommendations.

**Request:**
```json
{
  "customer_id": "customer_123",
  "article_ids": [108775015, 111565001, 111586001, ...]
}
```

**Response:** Same format as `/score`, but only returns top N articles by score.

## 🌐 Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test the API directly in your browser!

## 🔧 How It Works

1. **On Startup**: 
   - Loads trained Keras model
   - Loads preprocessor (StandardScaler + OneHotEncoder)
   - Precomputes user/item statistics from historical data

2. **On Request**:
   - Builds features for customer-article pairs
   - Preprocesses features using saved preprocessor
   - Runs neural network to predict scores
   - Returns probabilities (0-1 scale)

## 🎓 Learn More

- **Detailed Documentation**: See `API_README.md`
- **Interactive Demo**: Run `api_demo.ipynb` in Jupyter
- **Test Client**: See `test_api_client.py` for more examples

## 💡 Next Steps

1. **Test with real data**: Replace example customer/article IDs with real ones
2. **Integrate into your app**: Use the Python `requests` library or any HTTP client
3. **Deploy to production**: See `API_README.md` for Docker/Gunicorn deployment options
4. **Monitor performance**: Add logging and metrics as needed

## 🎉 You're Ready!

Your model is now accessible via a production-ready REST API. Start the server and begin making predictions!

---

**Need Help?**
- Check `API_README.md` for detailed documentation
- Run `python validate_api_setup.py` to check your setup
- All errors return descriptive JSON responses

