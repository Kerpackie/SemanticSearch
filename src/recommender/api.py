"""
FastAPI service for FFNN recommendation model.

Provides endpoints to:
- Score customer-product pairs
- Get recommendations for a customer given a list of products
"""

import os
import json
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import tensorflow as tf
from tensorflow import keras

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = "models/ffnn_20251016_154913"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_nn.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

TRANSACTIONS_PATH = "transactions_train.csv"
ARTICLES_PATH = "articles.csv"

ID_CUSTOMER = "customer_id"
ID_ITEM = "article_id"
DT_COL = "t_dat"
PRICE_COL = "price"
CAT_COL = "garment_group_name"

# ============================================
# LOAD MODEL AND RESOURCES
# ============================================
class ModelService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.meta = None
        self.transactions = None
        self.articles = None
        self.user_stats = None
        self.item_stats = None
        self.user_recency = None
        self.cutoff_date = None

    def load(self):
        """Load model, preprocessor, and precompute statistics from historical data."""
        print("Loading model and resources...")

        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        self.model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Loaded model from {MODEL_PATH}")

        # Load preprocessor
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Loaded preprocessor from {PREPROCESSOR_PATH}")

        # Load metadata
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                self.meta = json.load(f)
            print(f"✅ Loaded metadata from {META_PATH}")

        # Load historical data for feature engineering
        self._load_data()
        self._precompute_features()

        print("✅ Model service ready!")

    def _load_data(self):
        """Load transactions and articles data."""
        if os.path.exists(TRANSACTIONS_PATH):
            self.transactions = pd.read_csv(TRANSACTIONS_PATH)
            self.transactions[DT_COL] = pd.to_datetime(self.transactions[DT_COL])
            print(f"✅ Loaded {len(self.transactions):,} transactions")
        else:
            raise FileNotFoundError(f"Transactions file not found: {TRANSACTIONS_PATH}")

        if os.path.exists(ARTICLES_PATH):
            self.articles = pd.read_csv(ARTICLES_PATH)
            print(f"✅ Loaded {len(self.articles):,} articles")
        else:
            print("⚠️ Articles file not found - category features will be unavailable")
            self.articles = pd.DataFrame()

    def _precompute_features(self):
        """Precompute user and item statistics from historical data."""
        # Use the cutoff date from training if available
        if self.meta and self.meta.get("cutoff_date"):
            self.cutoff_date = pd.to_datetime(self.meta["cutoff_date"])
        else:
            # Default to max date in data
            self.cutoff_date = self.transactions[DT_COL].max()

        # Use only historical data before cutoff for feature computation
        hist = self.transactions[self.transactions[DT_COL] < self.cutoff_date].copy()

        # User average price
        self.user_stats = (
            hist.groupby(ID_CUSTOMER, as_index=False)[PRICE_COL]
            .mean()
            .rename(columns={PRICE_COL: "user_avg_price"})
        )

        # Item average price
        self.item_stats = (
            hist.groupby(ID_ITEM, as_index=False)[PRICE_COL]
            .mean()
            .rename(columns={PRICE_COL: "item_avg_price"})
        )

        # User recency (days since last purchase)
        user_last_purchase = (
            hist.groupby(ID_CUSTOMER, as_index=False)[DT_COL]
            .max()
            .rename(columns={DT_COL: "last_purchase_date"})
        )
        user_last_purchase["recency_days"] = (
            self.cutoff_date - user_last_purchase["last_purchase_date"]
        ).dt.days
        self.user_recency = user_last_purchase[[ID_CUSTOMER, "recency_days"]]

        print(f"✅ Precomputed features for {len(self.user_stats):,} users and {len(self.item_stats):,} items")

    def build_features(self, customer_id: str, article_ids: List[int]) -> pd.DataFrame:
        """Build feature dataframe for a customer and list of articles."""
        # Create candidate pairs
        pairs = pd.DataFrame({
            ID_CUSTOMER: [customer_id] * len(article_ids),
            ID_ITEM: article_ids
        })

        # Merge user features
        pairs = pairs.merge(self.user_stats, on=ID_CUSTOMER, how="left")

        # Merge item features
        pairs = pairs.merge(self.item_stats, on=ID_ITEM, how="left")

        # Merge recency
        pairs = pairs.merge(self.user_recency, on=ID_CUSTOMER, how="left")

        # Merge article category if available
        if CAT_COL and not self.articles.empty and CAT_COL in self.articles.columns:
            pairs = pairs.merge(
                self.articles[[ID_ITEM, CAT_COL]],
                on=ID_ITEM,
                how="left"
            )

        # Fill missing values with reasonable defaults
        if "user_avg_price" in pairs.columns:
            pairs["user_avg_price"] = pairs["user_avg_price"].fillna(
                self.user_stats["user_avg_price"].median()
            )

        if "item_avg_price" in pairs.columns:
            pairs["item_avg_price"] = pairs["item_avg_price"].fillna(
                self.item_stats["item_avg_price"].median()
            )

        if "recency_days" in pairs.columns:
            pairs["recency_days"] = pairs["recency_days"].fillna(
                self.user_recency["recency_days"].max() + 1
            )

        return pairs

    def predict(self, customer_id: str, article_ids: List[int]) -> np.ndarray:
        """
        Predict scores for customer-article pairs.

        Args:
            customer_id: Customer identifier
            article_ids: List of article identifiers

        Returns:
            Array of scores (probabilities) for each article
        """
        # Build features
        features_df = self.build_features(customer_id, article_ids)

        # Extract feature columns in the correct order
        if self.meta and self.meta.get("features"):
            feature_cols = self.meta["features"]
        else:
            # Fallback to what we know
            feature_cols = ["user_avg_price", "item_avg_price", "recency_days"]
            if CAT_COL and CAT_COL in features_df.columns:
                feature_cols.append(CAT_COL)

        # Ensure all required features are present
        for col in feature_cols:
            if col not in features_df.columns:
                # Add missing feature with default value
                if col == CAT_COL:
                    features_df[col] = "Unknown"
                else:
                    features_df[col] = 0.0

        X = features_df[feature_cols]

        # Preprocess
        X_processed = self.preprocessor.transform(X)

        # Predict
        scores = self.model.predict(X_processed, verbose=0).ravel()

        return scores


# Initialize service
service = ModelService()

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="FFNN Recommendation API",
    description="API for scoring customer-product pairs using a trained neural network",
    version="1.0.0"
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class ScoreRequest(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    article_ids: List[int] = Field(..., description="List of article identifiers to score")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f",
                "article_ids": [108775015, 111565001, 111586001]
            }
        }


class ScoredArticle(BaseModel):
    article_id: int
    score: float


class ScoreResponse(BaseModel):
    customer_id: str
    scores: List[ScoredArticle]

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f",
                "scores": [
                    {"article_id": 108775015, "score": 0.85},
                    {"article_id": 111565001, "score": 0.72},
                    {"article_id": 111586001, "score": 0.63}
                ]
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    metadata: Optional[Dict] = None


# ============================================
# ENDPOINTS
# ============================================
@app.on_event("startup")
async def startup_event():
    """Load model and resources on startup."""
    try:
        service.load()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if service.model is not None else "unhealthy",
        model_loaded=service.model is not None,
        model_path=MODEL_PATH,
        metadata=service.meta
    )


@app.post("/score", response_model=ScoreResponse)
async def score_articles(request: ScoreRequest):
    """
    Score articles for a customer.

    Returns a list of scores (probabilities) for each article in the input list.
    Higher scores indicate higher likelihood of purchase.
    """
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.article_ids:
        raise HTTPException(status_code=400, detail="article_ids cannot be empty")

    try:
        # Get predictions
        scores = service.predict(request.customer_id, request.article_ids)

        # Build response
        scored_articles = [
            ScoredArticle(article_id=article_id, score=float(score))
            for article_id, score in zip(request.article_ids, scores)
        ]

        return ScoreResponse(
            customer_id=request.customer_id,
            scores=scored_articles
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/recommend", response_model=ScoreResponse)
async def recommend_top_k(
    request: ScoreRequest,
    top_k: int = 12
):
    """
    Score articles and return top K recommendations.

    Returns the top K highest-scoring articles for the customer.
    """
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.article_ids:
        raise HTTPException(status_code=400, detail="article_ids cannot be empty")

    try:
        # Get predictions
        scores = service.predict(request.customer_id, request.article_ids)

        # Get top K indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build response with only top K
        scored_articles = [
            ScoredArticle(
                article_id=request.article_ids[idx],
                score=float(scores[idx])
            )
            for idx in top_indices
        ]

        return ScoreResponse(
            customer_id=request.customer_id,
            scores=scored_articles
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

