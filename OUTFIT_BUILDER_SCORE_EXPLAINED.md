# Outfit Builder Score Explanation

## 🎯 What is the Score?

The **score** you see in the Outfit Builder is a **weighted combination** of multiple signals:

### Score Composition

```
Final Score = Weighted Combination of:
1. Vector Similarity Score (from Mneme/Qdrant)
2. Cross-Encoder Reranking Score (from Arbiter)
3. Recommender Score (from Recommender service) - Optional
```

## 📊 Detailed Score Pipeline

### Step 1: Initial Vector Search (Mneme)
**Source**: Qdrant vector database  
**Score**: Cosine similarity between query embedding and product embedding  
**Range**: 0.0 to 1.0 (typically 0.3 - 0.95)

```
Query: "summer dress"
  ↓ (Glyph embedding)
  ↓ (Vector search in Mneme)
Product: "Light Blue Midi Dress" → Score: 0.87
```

### Step 2: Cross-Encoder Reranking (Arbiter)
**Source**: Cross-encoder model (better semantic understanding)  
**Score**: Relevance score from cross-encoder  
**Range**: 0.0 to 1.0

The cross-encoder looks at query + product pairs and gives a more accurate relevance score.

```
Query: "summer dress"
Product: "Light Blue Midi Dress"
  ↓ (Cross-encoder reranking)
Arbiter Score: 0.92 (more accurate than vector similarity)
```

### Step 3: Personalization (Recommender) - Optional
**Source**: Collaborative filtering model  
**Condition**: Only if `customerId` is provided  
**Score**: User preference score  
**Range**: 0.0 to 1.0

```
CustomerId: "user123"
Product: "Light Blue Midi Dress"
  ↓ (User preference model)
Recommender Score: 0.85 (this user likes similar items)
```

### Step 4: Final Score Calculation

**Code Location**: `SearchOrchestratorService.cs`, line ~267-276

```csharp
if (recommenderScores != null && recommenderScores.TryGetValue(arbiterResult.Id, out var recommenderScore))
{
    // Combine arbiter and recommender scores (60% arbiter, 40% recommender)
    finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore);
}
else
{
    // Use only arbiter score if no recommender score available
    finalScore = arbiterScore;
}
```

**Formula**:
- **With Recommender**: `Final = (0.6 × Arbiter) + (0.4 × Recommender)`
- **Without Recommender**: `Final = Arbiter`

## 📈 Example Score Flow

### Example 1: Guest User (No Personalization)
```
1. Vector Search (Mneme):     0.87
2. Reranking (Arbiter):        0.92
3. Recommender:                N/A (guest user)
4. Final Score:                0.92  ← Just Arbiter score
```

### Example 2: Logged-In User (With Personalization)
```
1. Vector Search (Mneme):     0.87
2. Reranking (Arbiter):        0.92
3. Recommender:                0.85
4. Final Score:                (0.6 × 0.92) + (0.4 × 0.85) = 0.892
   = 0.552 + 0.340 = 0.892
```

### Example 3: Strong Recommender Signal
```
1. Vector Search (Mneme):     0.75
2. Reranking (Arbiter):        0.80
3. Recommender:                0.95 (user loves this brand!)
4. Final Score:                (0.6 × 0.80) + (0.4 × 0.95) = 0.860
   = 0.480 + 0.380 = 0.860
```

Notice: Even though Arbiter score is lower, strong recommender signal boosts the final score!

## 🔍 Score Interpretation in Outfit Builder

In `GenerateReasoning()` (Program.cs ~539-545):

```csharp
if (result.Score > 0.9f)
    reasons.Add("Perfect match for your search");
else if (result.Score > 0.75f)
    reasons.Add("Great match for your style");
else if (result.Score > 0.6f)
    reasons.Add("Good alternative option");
```

### Score Ranges:
- **0.9 - 1.0**: Perfect match (very relevant to query AND user preferences)
- **0.75 - 0.9**: Great match (good relevance and/or personalization)
- **0.6 - 0.75**: Good alternative (decent match)
- **< 0.6**: Lower quality match (shown if needed to fill slots)

## 🎮 What Affects the Score?

### Arbiter Score (60% weight):
- ✅ Semantic relevance to the query
- ✅ Product name/description match
- ✅ Product type relevance
- ✅ Cross-encoder understanding

### Recommender Score (40% weight - if logged in):
- ✅ User's past purchase history
- ✅ Similar users' preferences
- ✅ Product popularity
- ✅ Collaborative filtering signals

### NOT Included:
- ❌ Price
- ❌ Stock availability
- ❌ Brand preference (unless learned by recommender)
- ❌ Product ratings

## 🔧 How It's Used in Outfit Builder

1. **Products are categorized** into slots (upper body, lower body, etc.)
2. **Within each slot**, products are sorted by score (descending)
3. **Top 10** highest-scoring products per slot are kept
4. **Score determines** the "reasoning" text shown to users
5. **Higher scores** appear first in the carousel

## 📝 Summary

**Question**: Is the score a total from recommender and cross encoder?

**Answer**: 
- ✅ **YES** - If you're logged in as a user with purchase history
- ❌ **NO** - If you're browsing as a guest

The score is:
- **Guest users**: 100% cross-encoder (Arbiter) score
- **Logged-in users**: 60% cross-encoder + 40% recommender score

This gives you the best of both worlds:
- **Arbiter** ensures relevance to the search query
- **Recommender** personalizes based on user preferences

The weighted combination (60/40 split) ensures that semantic relevance is prioritized, but personalization can still boost or lower scores based on user preferences.
