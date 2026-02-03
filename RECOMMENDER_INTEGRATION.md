# Recommender Integration Summary

## Overview
Successfully integrated the Python-based recommender service into the SemanticSearch Aspire application. The recommender is now part of the reranking pipeline in Nexus, combining its personalized scores with the Arbiter's relevance scores.

## Changes Made

### 1. Aspire AppHost Configuration (`src/SemanticSearch.AppHost/AppHost.cs`)
- Added recommender as an executable service using bash launcher script
- Configured HTTP endpoint on port 8000  
- Added recommender reference to Nexus service dependencies

### 2. Proto File Updates
#### `src/SemanticSearch.Nexus/Protos/search.proto`
- Added `optional string customer_id = 5` to `SearchRequest`
- Added `optional string customer_id = 4` to `TextSearchRequest`
- Added `optional string customer_id = 4` to `ImageSearchRequest`

#### `src/SemanticSearch.BFF/Protos/search.proto`
- Mirrored the same customer_id fields in BFF's proto file
- Ensures consistency across service boundaries

### 3. Nexus Service (`src/SemanticSearch.Nexus/`)
#### `Program.cs`
- Added HTTP client configuration for recommender service
- Reads service URL from Aspire configuration (`services:recommender:http:0`)
- Falls back to `http://localhost:8000` if not configured

#### `Services/SearchOrchestratorService.cs`
- Updated `Search()` method to pass `customer_id` to reranking
- Updated `SearchByText()` to forward `customer_id` from request
- **Completely rewrote `RerankResults()` method** with hybrid scoring:
  - Gets scores from both Arbiter (relevance) and Recommender (personalization)
  - Combines scores using weighted formula: `final_score = 0.6 * arbiter_score + 0.4 * recommender_score`
  - Falls back to Arbiter-only scoring when:
    - No `customer_id` is provided (user not logged in)
    - Recommender service is unavailable  
    - Recommender returns no scores
  - Re-ranks results based on final combined score

- Added `GetRecommenderScores()` helper method:
  - Calls recommender's `/score` endpoint
  - Extracts numeric article IDs from search results
  - Returns dictionary mapping article IDs to recommendation scores
  - Includes error handling and logging

- Added response model records for recommender API:
  - `RecommenderScoreResponse`
  - `ScoredArticle`

### 4. BFF Service (`src/SemanticSearch.BFF/Program.cs`)
- Updated `SearchRequest` record to include `optional string? CustomerId`
- Modified semantic search endpoint to pass `CustomerId` to Nexus gRPC call
- Set to empty string if not provided (proto requirement)

### 5. Recommender Service (`src/recommender/`)
- Created `run_recommender.sh` launch script:
  - Checks for Python3 availability
  - Auto-installs dependencies if uvicorn is missing
  - Reads PORT from environment variable
  - Launches FastAPI service using uvicorn

## Architecture Flow

```
Frontend (Search Request with optional customer_id)
    ↓
BFF (/api/search endpoint)
    ↓ (gRPC with customer_id)
Nexus (SearchOrchestratorService)
    ↓
1. NLP Processing (GptApi)
2. Text Embedding (Glyph)
3. Image Embedding (Eidolon - if image provided)
4. Vector Search (Mneme)
5. Reranking:
   a. Arbiter (relevance scoring)
   b. Recommender (personalization - if customer_id provided)
   c. Combine scores: 60% Arbiter + 40% Recommender
   d. Re-sort by final score
    ↓
Final ranked results returned to frontend
```

## Score Combination Logic

### When User is Logged In (customer_id provided):
```csharp
finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore)
```

### When User is NOT Logged In:
```csharp
finalScore = arbiterScore  // Only relevance-based ranking
```

## Configuration

### Recommender Service URL
The Nexus service discovers the recommender via Aspire service discovery:
- Primary: `builder.Configuration["services:recommender:http:0"]`
- Fallback: `http://localhost:8000`

### Score Weights
Currently hardcoded in `SearchOrchestratorService.cs`:
- Arbiter: 60% (relevance)
- Recommender: 40% (personalization)

**TODO**: Consider making these configurable via appsettings.json

## Dependencies

### Recommender Python Requirements
- fastapi>=0.104.0
- uvicorn[standard]>=0.24.0
- pydantic>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- tensorflow>=2.13.0
- joblib>=1.3.0

## Testing

### Testing Without User Login
```bash
curl -X POST http://localhost:5XXX/api/search \
  -H "Content-Type: application/json" \
  -d '{"Query": "black dress", "Limit": 10, "EnableReranking": true}'
```
Expected: Uses only Arbiter scores

### Testing With User Login
```bash
curl -X POST http://localhost:5XXX/api/search \
  -H "Content-Type: application/json" \
  -d '{"Query": "black dress", "Limit": 10, "EnableReranking": true, "CustomerId": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f"}'
```
Expected: Combines Arbiter and Recommender scores

## Known Issues & Next Steps

### 1. Proto Code Generation
The C# code generated from the proto files may need IDE/build tool refresh:
- Run `dotnet clean` then `dotnet build` on Nexus and BFF projects
- Restart IDE if CustomerId property is not recognized
- The proto field `customer_id` should generate as `CustomerId` in C#

### 2. Frontend Integration
Currently the frontend doesn't send `CustomerId`. Next steps:
- Add authentication/login to frontend
- Store customer ID in session/state
- Include customer ID in search requests

### 3. Score Weight Configuration
Consider making the score weights configurable:
```json
// appsettings.json
{
  "Reranking": {
    "ArbiterWeight": 0.6,
    "RecommenderWeight": 0.4
  }
}
```

### 4. Recommender Data Requirements
The recommender needs historical transaction data:
- `transactions_train.csv`
- `articles.csv`  
- Trained model files in `models/` directory
- Ensure these files exist in `src/recommender/` before running

### 5. Performance Considerations
- The recommender adds latency to search requests
- Consider caching recommender scores for frequent users
- Monitor impact on overall search performance

## Files Modified

- `src/SemanticSearch.AppHost/AppHost.cs`
- `src/SemanticSearch.Nexus/Protos/search.proto`
- `src/SemanticSearch.Nexus/Program.cs`
- `src/SemanticSearch.Nexus/Services/SearchOrchestratorService.cs`
- `src/SemanticSearch.BFF/Protos/search.proto`
- `src/SemanticSearch.BFF/Program.cs`

## Files Created

- `src/recommender/run_recommender.sh` (launch script)
