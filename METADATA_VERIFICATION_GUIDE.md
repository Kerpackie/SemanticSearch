# Metadata Verification Guide

## Problem Summary

The recommender system integration requires `article_id` in the metadata of search results, but the current implementation shows empty metadata. This document explains how to verify and fix the issue.

## Root Cause Analysis

The flow of data is:
1. **HmDataIngest** → Reads CSV, creates products with `article_id` in metadata → **Qdrant**
2. **Mneme** → Queries Qdrant, retrieves products with metadata → Returns via gRPC
3. **Nexus** → Receives products, maps to SearchResults → Should have metadata with `article_id`

If `article_id` is missing, the issue could be at any of these points.

## Diagnostic Tools Created

### 1. MnemeTest (Standalone Console App)

**Location**: `/src/MnemeTest/`

**Purpose**: Direct test of Mneme gRPC service to verify metadata retrieval.

**Usage**:
```bash
# Find Mneme port from Aspire dashboard, then:
cd src/MnemeTest
dotnet run <port>

# Example:
dotnet run 7123
```

**What it checks**:
- ✅ Can connect to Mneme gRPC service
- ✅ Can perform a vector search
- ✅ Metadata is present in response
- ✅ `article_id` is in metadata
- ✅ Counts how many products have/don't have `article_id`

**Output**: Detailed listing of first 5 products showing all metadata keys and values.

### 2. Enhanced MnemeTester (Aspire-integrated)

**Location**: `/src/R&D/MnemeTester/`

**Purpose**: HTTP API wrapper around Mneme for easy testing via browser/Postman.

**Changes Made**: Now includes metadata in response:
```json
{
  "id": "...",
  "name": "...",
  "description": "...",
  "score": 0.95,
  "metadata": {
    "article_id": "123456789",
    "color": "Blue",
    "type": "Shirts"
  },
  "hasArticleId": true,
  "articleId": "123456789"
}
```

**Usage**:
```bash
# Start via AppHost (automatically starts with Aspire)
# Then POST to http://localhost:<port>/products/search
# Body: { "textVector": [...], "imageVector": [...], "limit": 10 }
```

### 3. Enhanced Logging in Nexus

**Location**: `/src/SemanticSearch.Nexus/Services/SearchOrchestratorService.cs`

**Changes Made**:
- Counts results with/without `article_id`
- Logs first 3 results missing `article_id` with their available keys
- Provides helpful error message suggesting to run HmDataIngest

**Log Output**:
```
Metadata analysis: 0 results have article_id, 10 results missing article_id
Result a68e6e22-006f-cf86-9d2a-636b14fb4d8f missing article_id. Available keys: []
No valid article IDs found in metadata. All 10 results are missing article_id in metadata.
This suggests the Qdrant vector database needs to be re-indexed with metadata. Run HmDataIngest to populate article_id metadata.
```

### 4. Quick Test Script

**Location**: `/test-mneme-metadata.sh`

**Purpose**: Interactive script to help find Mneme port and run the test.

**Usage**:
```bash
./test-mneme-metadata.sh
```

## Verification Steps

### Step 1: Verify Data in Qdrant

First, check if the data was actually ingested with metadata:

1. Check HmDataIngest logs to confirm it completed successfully
2. Look for log entries showing metadata being added:
   ```
   { "article_id", row.ArticleId },
   { "color", row.Color },
   { "type", row.Type },
   ...
   ```

### Step 2: Test Mneme Service Directly

Run the standalone test:

```bash
# Start AppHost if not running
cd src/SemanticSearch.AppHost
dotnet run

# In another terminal, find Mneme port from Aspire dashboard, then:
cd src/MnemeTest
dotnet run <mneme-port>
```

**Expected Result**: All products should show metadata with `article_id`.

**If metadata is empty**:
- ❌ Data not in Qdrant → Re-run HmDataIngest
- ❌ Mneme not reading payload → Check ProductSearchService.cs

### Step 3: Test via MnemeTester HTTP API

Use the MnemeTester service (shown in Aspire dashboard):

```bash
curl -X POST http://localhost:<mneme-tester-port>/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "textVector": [0.1, 0.1, ...(768 values)],
    "limit": 5
  }'
```

Check response for `hasArticleId: true` and `articleId` value.

### Step 4: Test Full Flow via Nexus

Make a search request with a customer ID and check Nexus logs for the new diagnostic output.

## Common Issues and Solutions

### Issue 1: Empty Metadata

**Symptoms**: 
```
Result X missing article_id. Available keys: []
Metadata analysis: 0 results have article_id, 10 results missing article_id
```

**Solution**:
1. Re-run HmDataIngest to populate Qdrant with metadata
2. Verify HmDataIngest completed without errors
3. Check Qdrant is accessible and healthy

### Issue 2: Partial Metadata

**Symptoms**:
```
Metadata analysis: 3 results have article_id, 7 results missing article_id
```

**Solution**:
- HmDataIngest was interrupted or only partially completed
- Re-run HmDataIngest from start (or use StartFromIndex if resuming)

### Issue 3: Metadata Present but No article_id

**Symptoms**:
```
Result X missing article_id. Available keys: [name, description, color]
```

**Solution**:
- HmDataIngest ran before `article_id` was added to metadata
- Re-run HmDataIngest with current code

### Issue 4: Mneme Connection Failed

**Symptoms**: Test app can't connect to Mneme

**Solution**:
1. Ensure AppHost is running
2. Check Aspire dashboard for Mneme service status
3. Verify port number from dashboard
4. Check Mneme service logs for errors

## Code Changes Summary

### SearchOrchestratorService.cs
- ✅ Extract `article_id` from `result.Metadata["article_id"]` instead of `result.Id`
- ✅ Validate article IDs can be parsed as integers
- ✅ Map recommender scores back to result IDs using metadata lookup
- ✅ Enhanced logging for diagnostics

### MnemeTester/Program.cs
- ✅ Include metadata in HTTP response
- ✅ Add `HasArticleId` and `ArticleId` fields for easy verification

### New Files
- ✅ `/src/MnemeTest/` - Standalone diagnostic tool
- ✅ `/test-mneme-metadata.sh` - Helper script

## Next Steps

1. **Run the test**: Use `./test-mneme-metadata.sh` or manually test with MnemeTest
2. **Check results**: Verify all products have `article_id` in metadata
3. **If failing**: Re-run HmDataIngest to ensure proper data ingestion
4. **Verify fix**: Make a search request with customer ID and check Nexus logs

## Success Criteria

✅ MnemeTest shows: "✅ SUCCESS: All products have article_id in metadata!"
✅ Nexus logs show: "Metadata analysis: 10 results have article_id, 0 results missing article_id"
✅ Recommender scoring works: "Got recommender scores for 10 products"
✅ Final scores combine arbiter + recommender: "Product X: Arbiter=0.85, Recommender=0.92, Final=0.88"

