# Quick Reference: Outfit Builder Scores

## 🎯 The Score Formula

```
Guest User:     Score = Arbiter

Logged-In User: Score = (0.6 × Arbiter) + (0.4 × Recommender)
```

## 📊 What Each Component Means

| Component | What It Measures | Range | Weight |
|-----------|-----------------|-------|--------|
| **Arbiter** (Cross-Encoder) | Semantic relevance to query | 0.0 - 1.0 | 60% (or 100% for guests) |
| **Recommender** | User preference/history | 0.0 - 1.0 | 40% (logged-in only) |
| **Mneme** (Vector DB) | Initial similarity | 0.0 - 1.0 | Not used in final score |

## 💡 Quick Examples

### Example 1: Guest
```
Arbiter: 0.92
Final:   0.92  ← Just Arbiter
```

### Example 2: Logged-In (Low Recommender)
```
Arbiter:     0.92
Recommender: 0.50
Final:       (0.6×0.92) + (0.4×0.50) = 0.752
```

### Example 3: Logged-In (High Recommender)
```
Arbiter:     0.88
Recommender: 0.95
Final:       (0.6×0.88) + (0.4×0.95) = 0.908
```

## 📍 Where to Find in Code

| Location | File | Line |
|----------|------|------|
| **Score Fusion** | `SearchOrchestratorService.cs` | ~267-276 |
| **Reasoning Text** | `Program.cs` (BFF) | ~539-545 |
| **Arbiter Call** | `SearchOrchestratorService.cs` | ~235-249 |
| **Recommender Call** | `SearchOrchestratorService.cs` | ~306-370 |

## 🎨 Score Interpretation

| Range | Reasoning Text | Meaning |
|-------|---------------|---------|
| > 0.9 | "Perfect match for your search" | Highly relevant AND personalized |
| 0.75 - 0.9 | "Great match for your style" | Good relevance or personalization |
| 0.6 - 0.75 | "Good alternative option" | Decent match |
| < 0.6 | (Default) "Recommended for you" | Lower quality match |

## 🔍 Debugging Score Issues

**If scores seem wrong:**

1. Check if user is logged in (impacts weight)
2. Check Arbiter logs for cross-encoder scores
3. Check Recommender logs for personalization scores
4. Verify the 60/40 weighting is applied correctly

**Logs to look for:**
```
Product {Id}: Arbiter={ArbiterScore:F3}, Recommender={RecommenderScore:F3}, Final={FinalScore:F3}
```

## 📚 Related Documentation

- [OUTFIT_BUILDER_SCORE_EXPLAINED.md](./OUTFIT_BUILDER_SCORE_EXPLAINED.md) - Full explanation
- [OUTFIT_BUILDER_SCORE_FLOW.md](./OUTFIT_BUILDER_SCORE_FLOW.md) - Visual flow diagram
- [OUTFIT_BUILDER_ITEMS_DEBUG.md](./OUTFIT_BUILDER_ITEMS_DEBUG.md) - Debug guide
