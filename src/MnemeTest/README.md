# Mneme Metadata Test

This is a diagnostic tool to verify that the Mneme service is correctly retrieving metadata (especially `article_id`) from Qdrant.

## Purpose

When search results don't have `article_id` in their metadata, the recommender system can't score them. This tool helps diagnose whether:
1. The data in Qdrant has the metadata
2. Mneme is correctly retrieving it from Qdrant
3. The metadata is being properly serialized through gRPC

## Usage

1. **Start the AppHost** (if not already running):
   ```bash
   cd ../SemanticSearch.AppHost
   dotnet run
   ```

2. **Find the Mneme port** from the Aspire dashboard (usually shown in console output or at http://localhost:15888)

3. **Run this test**:
   ```bash
   dotnet run <mneme-port>
   ```
   
   Example:
   ```bash
   dotnet run 7123
   ```

## Expected Output

✅ **Success**: All products should have `article_id` in their metadata.

Example:
```
Metadata (5 entries):
  name: Casual Shirt
  description: ...
  article_id: 123456789
  color: Blue
  type: Shirts
  ✅ article_id present: 123456789
```

❌ **Failure**: If metadata is empty or missing `article_id`:
```
Metadata (0 entries):
  ⚠️  NO METADATA FOUND!
```

This means you need to re-run HmDataIngest to populate Qdrant with proper metadata.

## What This Tests

1. ✅ Connection to Mneme gRPC service
2. ✅ Basic vector search functionality
3. ✅ Metadata retrieval from Qdrant
4. ✅ Metadata serialization through gRPC
5. ✅ Presence of `article_id` field in metadata

## Troubleshooting

**Can't connect to Mneme**:
- Make sure the AppHost is running
- Check the Aspire dashboard for the actual port
- Verify Mneme service is healthy (green in dashboard)

**No metadata returned**:
- Run HmDataIngest to re-index products with metadata
- Check Qdrant is running and accessible
- Verify data was ingested correctly

**Some products have metadata, some don't**:
- Partial ingestion - run HmDataIngest to completion
- Check HmDataIngest logs for errors
