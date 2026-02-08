# Outfit Builder - "Only Showing 1-2 Items" Debug Guide

## 🎯 Score Explanation

**Q: What is the score shown for each product?**

**A**: It's a weighted combination:
- **Guest users**: 100% Cross-Encoder (Arbiter) score
- **Logged-in users**: `(0.6 × Arbiter) + (0.4 × Recommender)`

See [OUTFIT_BUILDER_SCORE_EXPLAINED.md](./OUTFIT_BUILDER_SCORE_EXPLAINED.md) for full details.

## 🔍 Issue Description
The outfit builder is working but some slots only show 1-2 products instead of the expected 10.

## ✅ Fixes Applied

### 1. **Use Article ID in Recommendations** (Line ~356)
**Problem**: Recommendations were using document UUIDs as IDs instead of article IDs.

**Before**:
```csharp
var recommendation = new RecommendationDto(
    result.Id, // This is a UUID!
    ...
);
```

**After**:
```csharp
var recommendation = new RecommendationDto(
    articleId, // Now using the actual article ID
    ...
);
```

**Impact**: Frontend can now properly identify and display products.

### 2. **Enhanced Logging Throughout the Stack**

#### Backend Logs (C# Console):
- Shows article IDs being sent to PgApi
- Shows how many products PgApi returns
- Shows slot breakdown BEFORE filtering
- Shows first and last item in each slot with scores
- Shows filtering results (kept X of Y items)

#### Frontend Logs (Browser Console):
- API response structure
- Each slot's recommendation count
- First recommendation in each slot
- OutfitSlot component receives data
- SlotCarousel component receives recommendations
- All recommendation IDs being rendered

## 🔍 What to Check in Logs

### Backend Console - Look For:
```
PgApi returned 100 full products  ← Should be high number

[CategorizeIntoSlots] Slot breakdown BEFORE filtering:
  - full_body: 25 items
    First item: Summer Dress (Score: 0.95)
    Last item: Casual Dress (Score: 0.72)
  - accessories: 15 items
    First item: Handbag (Score: 0.88)
    Last item: Belt (Score: 0.65)

[CategorizeIntoSlots] full_body: kept 10 of 25 items (top 10)  ← Should be 10
[CategorizeIntoSlots] accessories: kept 10 of 15 items (top 10)  ← Should be 10
```

### Browser Console - Look For:
```
[API] Slot "full_body": {
  slotType: "full_body",
  recommendationsCount: 10,  ← Should be 10!
  reasoning: "..."
}
[API]   First recommendation: {id: "123456", name: "...", score: 0.95}

[OutfitSlot] full_body: {
  itemCount: 10,  ← Should match!
  ...
}

[SlotCarousel] Rendering with 10 recommendations  ← Should be 10!
```

## 🐛 Possible Causes

### If Backend Shows 10 But Frontend Shows Less:

1. **Data Transformation Issue**
   - Check if `data.slots.slotName.recommendations` is an array
   - Check if recommendations have valid IDs

2. **Type Mismatch**
   - Backend sends `Recommendations` (capital R)
   - Frontend expects `recommendations` (lowercase r)
   - Check JSON serialization settings

3. **Filtering in Frontend**
   - Check if OutfitBuilder is filtering slots
   - Check if SlotCarousel has any display logic

### If Backend Shows Less Than 10:

1. **Not Enough Products Categorized**
   - Check categorization rules
   - Some products might not match any slot

2. **Duplicate Filtering**
   - Check if products are being deduplicated somewhere

## 🚀 Next Steps

1. **Search** for "summer dress" or "casual outfit"
2. **Click** "👔 Outfit Builder"
3. **Check BOTH consoles**:
   - Backend: How many items in each slot?
   - Frontend: How many items received?
4. **Share the logs** showing:
   - Slot breakdown BEFORE filtering
   - Kept X of Y items (top 10)
   - API response slot counts
   - SlotCarousel rendering counts

This will pinpoint exactly where the items are getting lost! 🎯
