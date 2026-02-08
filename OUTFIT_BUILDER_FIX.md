# Outfit Builder Debug Guide

## 🔍 Root Cause Identified!

**The Problem**: PgApi returned ZERO products for all 100 search results.

**Why**: Nexus returns search results with document UUIDs as IDs (e.g., `672f01a2-8e4a-0453-d27f-9a4ae767e3c2`), but PgApi expects actual article IDs (the product SKUs from H&M data).

**The Fix**: Extract the actual `article_id` from the search result metadata instead of using the document UUID.

### Before (Broken):
```csharp
var articleIds = grpcResponse.Results.Select(r => r.Id).ToList();
// r.Id = UUID from vector DB document
// PgApi can't find products with these UUIDs
```

### After (Fixed):
```csharp
var articleIds = grpcResponse.Results
    .Select(r => {
        // Try to get article_id from metadata first
        if (r.Metadata.TryGetValue("article_id", out var articleId))
        {
            return articleId;
        }
        // Fallback to using the ID directly
        return r.Id;
    })
    .ToList();
// Now using actual article IDs from metadata
// PgApi can find the products!
```

## 🔍 Comprehensive Logging Added

I've added extensive logging throughout the entire outfit builder flow to help diagnose the issue.

## Where to Look for Logs

### Backend Logs (C# Console)

Look for these log messages when you click the Outfit Builder button:

```
=== OUTFIT SEARCH STARTED ===
Query: <your query>, CustomerId: <customer id>
Calling Nexus with limit: 100
Nexus returned <X> results, TotalResults: <Y>
Article IDs to fetch: <X>
Fetching product details from PgApi for <X> articles
PgApi returned <X> full products
Starting categorization into slots...
```

Then in the categorization function:

```
[CategorizeIntoSlots] Starting with X results and Y products
[CategorizeIntoSlots] Product 1: ID=<article_id>
  - Name: <product name>
  - ProductGroup: '<value>'
  - ProductType: '<value>'
  - GarmentGroup: '<value>'
  -> Categorized as: <SLOT_NAME> or NOT CATEGORIZED!
```

After processing:

```
[CategorizeIntoSlots] Processed X products, categorized Y
[CategorizeIntoSlots] Slot breakdown BEFORE filtering:
  - upper_body: X items
  - lower_body: X items
  ...
[CategorizeIntoSlots] upper_body: kept 10 of X items (top 10)
[CategorizeIntoSlots] Returning X non-empty slots
```

Finally:

```
=== OUTFIT SEARCH COMPLETED === Total slots: X
```

### Frontend Logs (Browser Console)

When you click the Outfit Builder button:

```
🎨 [OUTFIT BUILDER] Button clicked!
🎨 [OUTFIT BUILDER] Current viewMode: grid
🎨 [OUTFIT BUILDER] Search query: <your query>
🎨 [OUTFIT BUILDER] Customer ID: <customer id>
🎨 [OUTFIT BUILDER] Calling outfitSearch API...
🎨 [OUTFIT BUILDER] Request params: {query: "...", customerId: "..."}
```

API call logs:

```
[API] outfitSearch called with: {query: "...", customerId: "..."}
[API] Request body: {query: "..."}
[API] Sending POST to: /api/outfit-search
[API] Response status: 200
[API] Response ok: true
[API] Response data: {slots: {...}, totalResults: 100, ...}
```

Response processing:

```
🎨 [OUTFIT BUILDER] API Response received:
🎨 [OUTFIT BUILDER] - Total results: 100
🎨 [OUTFIT BUILDER] - Processed query: "..."
🎨 [OUTFIT BUILDER] - Slots: {upper_body: {...}, ...}
🎨 [OUTFIT BUILDER] - Number of slots: X
🎨 [OUTFIT BUILDER]   - upper_body: X items
🎨 [OUTFIT BUILDER]   - lower_body: X items
...
🎨 [OUTFIT BUILDER] State updated successfully
🎨 [OUTFIT BUILDER] Loading complete
```

Rendering decision:

```
[App] Outfit mode rendering decision: {
  viewMode: "outfit",
  searchQuery: "...",
  loading: false,
  outfitSlots: {...},
  outfitSlotsKeys: ["upper_body", "lower_body", ...]
}
[App] -> Rendering OutfitBuilder with slots
```

OutfitBuilder component:

```
[OutfitBuilder] Rendering with: {
  slotsProvided: true,
  slotKeys: ["upper_body", "lower_body", ...],
  loading: false
}
[OutfitBuilder] Available slots: ["upper_body", "lower_body", ...]
[OutfitBuilder] Available slot count: X
```

## 🔍 What to Check

### If you see "No slots returned!" warning:
1. Check the backend logs for categorization details
2. Look at the first 5 products logged - what are their ProductGroup, ProductType, and GarmentGroup values?
3. Are they matching any of our categorization rules?

### If products are "NOT CATEGORIZED":
The logs will show you the exact values of ProductGroup, ProductType, and GarmentGroup for the first 5 products. This will tell us what keywords we need to add to catch them.

### If slots are empty after filtering:
Check if products are being categorized but then removed. The logs show counts before and after filtering.

## 🚀 How to Test

1. **Start backend** (make sure console output is visible)
2. **Start frontend** (open browser console)
3. **Search** for something like "summer dress"
4. **Click** the "👔 Outfit Builder" button
5. **Watch both consoles** for the log messages
6. **Copy the logs** and share them to diagnose the exact issue

## Example of What You Might See

If categorization is failing, you might see:

```
[CategorizeIntoSlots] Product 1: ID=123456
  - Name: "Summer Dress"
  - ProductGroup: 'garment full body'    ← Contains "full" ✓
  - ProductType: 'dress'                  ← Contains "dress" ✓
  - GarmentGroup: 'dresses'               ← Contains "dress" ✓
  -> Categorized as: FULL_BODY            ← SUCCESS!
```

Or if failing:

```
[CategorizeIntoSlots] Product 1: ID=123456
  - Name: "Mystery Item"
  - ProductGroup: 'something_weird'       ← No match ✗
  - ProductType: 'unknown_type'           ← No match ✗
  - GarmentGroup: 'misc'                  ← No match ✗
  -> NOT CATEGORIZED!                     ← PROBLEM!
```

This will tell us exactly what keywords to add!
