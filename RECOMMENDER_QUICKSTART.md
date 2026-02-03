# Quick Start: Recommender Integration

## Prerequisites

1. **Install Python dependencies**:
   ```bash
   cd src/recommender
   pip3 install -r requirements_api.txt
   ```

2. **Ensure recommender model and data files exist**:
   - `src/recommender/models/ffnn_*/model.keras`
   - `src/recommender/models/ffnn_*/preprocessor_nn.joblib`
   - `src/recommender/transactions_train.csv`
   - `src/recommender/articles.csv`

## Running the Application

### Option 1: Via Aspire AppHost (Recommended)
```bash
cd src/SemanticSearch.AppHost
dotnet run
```

The recommender will automatically start as part of the Aspire orchestration.

### Option 2: Run Recommender Standalone
```bash
cd src/recommender
./run_recommender.sh
# Or directly with uvicorn:
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Usage

### Search WITHOUT Personalization (No Login)
```bash
POST /api/search
{
  "Query": "summer dress",
  "Limit": 20,
  "EnableReranking": true
}
```
Result: Pure relevance-based ranking from Arbiter

### Search WITH Personalization (User Logged In)
```bash
POST /api/search
{
  "Query": "summer dress",
  "Limit": 20,
  "EnableReranking": true,
  "CustomerId": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f"
}
```
Result: Hybrid ranking (60% Arbiter + 40% Recommender)

## Verifying the Integration

### 1. Check Recommender Health
```bash
curl http://localhost:8000/
```
Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/ffnn_*/model.keras",
  "metadata": {...}
}
```

### 2. Test Recommender Directly
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f",
    "article_ids": [108775015, 111565001, 111586001]
  }'
```

### 3. Monitor Logs
Watch Nexus logs for recommender integration:
```bash
# Look for messages like:
"Got recommender scores for {Count} products"
"Product {Id}: Arbiter={ArbiterScore}, Recommender={RecommenderScore}, Final={FinalScore}"
```

## Troubleshooting

### Recommender Not Starting
- Check Python3 is installed: `python3 --version`
- Install dependencies: `pip3 install -r src/recommender/requirements_api.txt`
- Check port 8000 is available: `lsof -i :8000`

### Scores Not Combining
- Verify CustomerId is being passed to BFF
- Check Nexus logs for recommender connection errors
- Ensure article IDs are numeric and valid

### Performance Issues
- Recommender adds ~50-200ms latency per request
- Consider reducing number of products sent to recommender
- Cache recommender scores for frequent users

## Score Weight Tuning

Current weights (in `SearchOrchestratorService.cs`):
```csharp
finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore)
```

To adjust:
1. Edit line ~237 in `SearchOrchestratorService.cs`
2. Rebuild Nexus: `dotnet build src/SemanticSearch.Nexus`
3. Restart application

Recommended ranges:
- More relevance-focused: 70/30 or 80/20 (Arbiter/Recommender)
- More personalization-focused: 50/50 or 40/60
- Current balanced: 60/40

## Next Steps

1. **Add Frontend Authentication**: Capture and send customer_id from UI
2. **Performance Monitoring**: Add metrics for recommender latency and score impact
3. **A/B Testing**: Compare search quality with/without recommender
4. **Score Weight Configuration**: Move weights to appsettings.json
5. **Caching**: Cache recommender scores to reduce latency
