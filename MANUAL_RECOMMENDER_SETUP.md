# Manual Recommender Setup Guide

## Quick Start

The recommender service now runs **manually** outside of Aspire to avoid TensorFlow compatibility issues.

### Step 1: Start the Recommender Service

Open a **separate terminal** and run:

```bash
cd /Users/kerpackie/RiderProjects/SemanticSearch/src/recommender
python3 api.py
```

You should see:
```
============================================================
Starting FFNN Recommendation API on port 8000
============================================================
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Verify It's Running

In another terminal, test the health endpoint:

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/ffnn_20251016_154913/model.keras",
  "metadata": {...}
}
```

### Step 3: Start Aspire Application

In your main terminal:

```bash
cd /Users/kerpackie/RiderProjects/SemanticSearch/src/SemanticSearch.AppHost
dotnet run
```

This will start all other services and automatically connect to the recommender at `http://localhost:8000`.

### Step 4: Start Frontend (Optional)

In another terminal:

```bash
cd /Users/kerpackie/RiderProjects/SemanticSearch/src/semantic-search-frontned
npm run dev
```

Open: `http://localhost:5173`

---

## Configuration

The Nexus service is configured to look for the recommender at:
- **Default:** `http://localhost:8000` (hardcoded fallback)
- **Override:** Set `services:recommender:http:0` in appsettings if needed

If the recommender is not running, Nexus will gracefully fall back to Arbiter-only ranking.

---

## Troubleshooting

### Recommender won't start

**Issue:** TensorFlow segmentation fault

**Solution:** Try these in order:

1. **Use a virtual environment:**
   ```bash
   cd src/recommender
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_api.txt
   python3 api.py
   ```

2. **Try tensorflow-macos (Apple Silicon):**
   ```bash
   pip3 uninstall tensorflow
   pip3 install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0
   ```

3. **Try older TensorFlow:**
   ```bash
   pip3 install tensorflow==2.12.0
   ```

4. **Use conda instead:**
   ```bash
   conda create -n recommender python=3.10
   conda activate recommender
   pip install -r requirements_api.txt
   python3 api.py
   ```

### Port 8000 already in use

**Check what's using it:**
```bash
lsof -i :8000
```

**Kill it or change port:**
```bash
# In api.py, change the port at the bottom:
port = int(os.environ.get("PORT", 8001))  # Use 8001 instead

# Then update Nexus appsettings.json:
"services": {
  "recommender": {
    "http": ["http://localhost:8001"]
  }
}
```

### Nexus can't connect to recommender

**Check Nexus logs for:**
```
Failed to get recommender scores, falling back to arbiter-only scoring
```

This is **normal** and graceful - search still works, just without personalization.

**To fix:**
1. Verify recommender is running: `curl http://localhost:8000/`
2. Check Nexus appsettings for correct URL
3. Check firewall isn't blocking localhost connections

---

## Testing the Integration

### Test 1: Without Recommender (Arbiter Only)

1. **Don't start** the recommender
2. Start Aspire and frontend
3. Search as "Emma Johnson"
4. Check Nexus logs - should see "falling back to arbiter-only"
5. Results are purely relevance-based

### Test 2: With Recommender (Hybrid)

1. **Start** the recommender: `cd src/recommender && python3 api.py`
2. Start Aspire and frontend
3. Search as "Emma Johnson"
4. Check Nexus logs - should see "Got recommender scores for X products"
5. Results combine relevance + personalization

### Test 3: Compare Results

1. Search "summer dress" as Guest → Note top 3 results
2. Search "summer dress" as Emma → Compare top 3 results
3. Results should differ if recommender is running
4. Switch to Michael → Results should differ again

---

## Process Overview

**Terminal 1: Recommender**
```bash
cd src/recommender
python3 api.py
# Keep running
```

**Terminal 2: Aspire**
```bash
cd src/SemanticSearch.AppHost
dotnet run
# Keep running
```

**Terminal 3: Frontend**
```bash
cd src/semantic-search-frontned
npm run dev
# Keep running
```

**Browser:** Open `http://localhost:5173`

---

## Why Manual Launch?

The recommender service has TensorFlow dependencies that can cause segmentation faults when launched via Aspire's process management on some macOS configurations. Running it manually:

1. ✅ Easier to debug TensorFlow issues
2. ✅ Can use virtual environments
3. ✅ Better control over Python version
4. ✅ Can restart independently
5. ✅ See full error messages

The application still works seamlessly - Nexus connects automatically to `localhost:8000`.

---

## Success Indicators

✅ Recommender starts without segfault  
✅ Health endpoint returns 200 OK  
✅ Aspire starts all other services  
✅ Nexus logs show "Got recommender scores"  
✅ Frontend user selection works  
✅ Results differ between users  

**You're all set!** 🎉
