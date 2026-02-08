# Outfit Builder - Score Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    OUTFIT BUILDER SCORE FLOW                     │
└─────────────────────────────────────────────────────────────────┘

User Query: "summer dress"
     │
     ▼
┌─────────────────────┐
│  1. NLP Processing  │  GptApi → "Light Beige Linen Midi Dress"
│     (GptApi)        │          "Neutral Espadrille Wedges"
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Text Embedding  │  Glyph → [0.23, 0.45, ..., 0.89]
│     (Glyph)         │          (384-dimensional vector)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Vector Search   │  Mneme/Qdrant → 100 products
│     (Mneme)         │  Each with cosine similarity score
└──────────┬──────────┘  Example: 0.87, 0.85, 0.83, ...
           │
           ├──────────────────────────────────────┐
           │                                      │
           ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  4a. Cross-Encoder  │              │  4b. Recommender    │
│      Reranking      │              │   Personalization   │
│     (Arbiter)       │              │   (Recommender)     │
│                     │              │                     │
│ Re-scores based on  │              │ Scores based on:    │
│ semantic relevance  │              │ - User history      │
│                     │              │ - Similar users     │
│ Score: 0.92         │              │ - Collaborative     │
└──────────┬──────────┘              └──────────┬──────────┘
           │                                    │
           │                                    │ (Optional)
           │                                    │ Only if logged in
           │                                    │
           └──────────────┬─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  5. Score Fusion    │
              │                     │
              │  IF logged in:      │
              │  Final = (0.6 × A)  │
              │        + (0.4 × R)  │
              │                     │
              │  IF guest:          │
              │  Final = A          │
              │                     │
              │  Example (logged):  │
              │  (0.6×0.92)         │
              │  + (0.4×0.85)       │
              │  = 0.892            │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 6. Categorization   │
              │                     │
              │ Products → Slots:   │
              │ - Upper Body        │
              │ - Lower Body        │
              │ - Full Body         │
              │ - Shoes             │
              │ - Accessories       │
              │ - Underwear         │
              │ - Swimwear          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 7. Top-K Filtering  │
              │                     │
              │ Per slot:           │
              │ Sort by score DESC  │
              │ Take top 10         │
              │                     │
              │ full_body: 10/25    │
              │ accessories: 10/15  │
              │ shoes: 10/12        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 8. Carousel Display │
              │                     │
              │ Each slot shows:    │
              │ [Item 1] [Item 2]   │
              │ [Item 3] ...        │
              │                     │
              │ With reasoning:     │
              │ "Perfect match"     │
              │ (score > 0.9)       │
              └─────────────────────┘
```

## Score Weight Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                     SCORE COMPOSITION                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GUEST USER (No CustomerId):                                │
│  ┌──────────────────────────────────────┐                   │
│  │                                      │                   │
│  │   Arbiter Score (Cross-Encoder)      │                   │
│  │             100%                     │                   │
│  │                                      │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│  LOGGED-IN USER (With CustomerId):                          │
│  ┌──────────────┬──────────────────────┐                   │
│  │              │                      │                   │
│  │   Arbiter    │    Recommender       │                   │
│  │     60%      │        40%           │                   │
│  │              │                      │                   │
│  │  Semantic    │   Personalization    │                   │
│  │  Relevance   │   User Preference    │                   │
│  │              │                      │                   │
│  └──────────────┴──────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Example Scenarios

### Scenario 1: Guest User
```
Query: "summer dress"
Product: "Floral Midi Dress"

Mneme (vector):     0.85  (not used directly)
Arbiter:            0.92  ←─┐
Recommender:        N/A     │
                            │
Final Score:        0.92  ←─┘
Reasoning:         "Perfect match for your search"
```

### Scenario 2: New User (Logged In, No History)
```
Query: "summer dress"
Product: "Floral Midi Dress"

Mneme (vector):     0.85  (not used directly)
Arbiter:            0.92  ←─┐ 60%
Recommender:        0.50  ←─┘ 40% (low - no history)
                            
Final Score:        (0.6×0.92) + (0.4×0.50) = 0.752
Reasoning:         "Great match for your style"
```

### Scenario 3: Returning User (Strong Preferences)
```
Query: "summer dress" 
Product: "Floral Midi Dress" (user has bought similar before)

Mneme (vector):     0.85  (not used directly)
Arbiter:            0.88  ←─┐ 60%
Recommender:        0.95  ←─┘ 40% (high - user loves this style!)
                            
Final Score:        (0.6×0.88) + (0.4×0.95) = 0.908
Reasoning:         "Perfect match for your search"
```

Notice: Recommender can boost the score even if Arbiter score is lower!

## Key Takeaways

1. **Arbiter (60%)** = "Is this relevant to the search?"
2. **Recommender (40%)** = "Would THIS user like it?"
3. **Mneme score** is not used directly (only for initial ranking)
4. **Final score** determines carousel order within each slot
5. **Top 10** per slot are shown (sorted by final score)
