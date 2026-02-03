"""
Test client for the FFNN Recommendation API.

Demonstrates how to interact with the API endpoints.
"""

import requests
import json


# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_score_articles(customer_id: str, article_ids: list):
    """Test the score endpoint."""
    print("=" * 60)
    print("Testing Score Endpoint")
    print("=" * 60)

    payload = {
        "customer_id": customer_id,
        "article_ids": article_ids
    }

    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")

    response = requests.post(f"{BASE_URL}/score", json=payload)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nCustomer: {result['customer_id']}")
        print(f"\nScores:")
        for item in result['scores']:
            print(f"  Article {item['article_id']}: {item['score']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()


def test_recommend_top_k(customer_id: str, article_ids: list, top_k: int = 5):
    """Test the recommend endpoint."""
    print("=" * 60)
    print(f"Testing Recommend Endpoint (Top {top_k})")
    print("=" * 60)

    payload = {
        "customer_id": customer_id,
        "article_ids": article_ids
    }

    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")

    response = requests.post(f"{BASE_URL}/recommend?top_k={top_k}", json=payload)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nCustomer: {result['customer_id']}")
        print(f"\nTop {top_k} Recommendations:")
        for i, item in enumerate(result['scores'], 1):
            print(f"  {i}. Article {item['article_id']}: {item['score']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    # Example customer and articles
    # Replace these with actual IDs from your dataset
    EXAMPLE_CUSTOMER = "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f"
    EXAMPLE_ARTICLES = [
        108775015, 111565001, 111586001, 372860001,
        464131002, 541518023, 610776002, 706016001,
        751471001, 759871002, 866731001, 918292001
    ]

    try:
        # Test health check
        test_health_check()

        # Test scoring all articles
        test_score_articles(EXAMPLE_CUSTOMER, EXAMPLE_ARTICLES)

        # Test getting top recommendations
        test_recommend_top_k(EXAMPLE_CUSTOMER, EXAMPLE_ARTICLES, top_k=5)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the server is running with: python api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

