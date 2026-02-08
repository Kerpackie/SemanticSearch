# 🎨 Outfit Builder - Visual Guide

## UI Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SEARCH BAR                                │
│  "casual summer outfit"                      [Search Button] │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Search results for "casual summer outfit"                   │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ 📋 Grid View │  │👔 Outfit     │ ← Toggle Buttons        │
│  │   (active)   │  │   Builder    │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  [Click Outfit Builder]
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            🎨 Your Personalized Outfit                       │
│  We've categorized 4 style slots for you.                   │
│  Scroll through each to find your perfect match!            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  👕 Upper Body                            12 items      ▼   │ ← Collapsed Slot
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ +8                           │
│  │img │ │img │ │img │ │img │                               │
│  └────┘ └────┘ └────┘ └────┘                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    [Click to Expand]
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  👕 Upper Body                            12 items      ▲   │ ← Expanded Slot
├─────────────────────────────────────────────────────────────┤
│  💡 Why this slot? Items for your upper body like shirts,   │
│     tops, jackets, and sweaters                             │
├─────────────────────────────────────────────────────────────┤
│  ◄                    CAROUSEL                          ►   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │   #1     │ │   #2     │ │   #3     │                    │
│  │  95%     │ │  92%     │ │  89%     │                    │
│  ├──────────┤ ├──────────┤ ├──────────┤                    │
│  │          │ │          │ │          │                    │
│  │  [IMG]   │ │  [IMG]   │ │  [IMG]   │                    │
│  │          │ │          │ │          │                    │
│  ├──────────┤ ├──────────┤ ├──────────┤                    │
│  │Cotton    │ │Linen     │ │Summer    │                    │
│  │T-Shirt   │ │Blouse    │ │Tank Top  │                    │
│  ├──────────┤ ├──────────┤ ├──────────┤                    │
│  │💡 Why?   │ │💡 Why?   │ │💡 Why?   │                    │
│  │Perfect   │ │Great     │ │Great     │                    │
│  │match, in │ │match, in │ │match, in │                    │
│  │blue      │ │white     │ │pink      │                    │
│  ├──────────┤ ├──────────┤ ├──────────┤                    │
│  │🎨 Blue   │ │🎨 White  │ │🎨 Pink   │                    │
│  │📦 T-Shirt│ │📦 Blouse │ │📦 Tank   │                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
│  ● ● ○ ○ ○ ○ ○ ○ ○ ○  ← Carousel Indicators              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  👖 Lower Body                            8 items       ▼   │
│  [Preview thumbnails...]                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  👜 Accessories                           6 items       ▼   │
│  [Preview thumbnails...]                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  👟 Footwear                              10 items      ▼   │
│  [Preview thumbnails...]                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  💡 Tip: Each slot shows the top 10 re-ranked items based   │
│     on your preferences and the search query.               │
└─────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

```
App
 └── OutfitBuilder
      ├── OutfitSlot (Upper Body)
      │    └── SlotCarousel
      │         ├── RecommendationCard (#1)
      │         ├── RecommendationCard (#2)
      │         ├── RecommendationCard (#3)
      │         └── ... (up to 10)
      │
      ├── OutfitSlot (Lower Body)
      │    └── SlotCarousel
      │         └── ...
      │
      ├── OutfitSlot (Accessories)
      │    └── SlotCarousel
      │         └── ...
      │
      └── OutfitSlot (Footwear)
           └── SlotCarousel
                └── ...
```

## Data Flow

```
User Search Query
      ↓
Frontend: outfitSearch(query, customerId)
      ↓
BFF: /api/outfit-search
      ↓
Nexus: SearchByTextAsync(query, limit=100)
      ↓
Mneme: Vector Search
      ↓
Arbiter: LLM Re-ranking
      ↓
Recommender: Personalization (if logged in)
      ↓
BFF: Get full product details from PgApi
      ↓
BFF: CategorizeIntoSlots()
      ├── Check product group
      ├── Check product type
      ├── Assign to appropriate slot
      ├── Generate reasoning
      └── Keep top 10 per slot
      ↓
Frontend: Render OutfitBuilder
      ├── Show slots with preview
      ├── Expand slot on click
      ├── Render carousel
      └── Show reasoning
```

## User Interaction Flow

```
1. SEARCH
   User: "summer beach outfit"
   ↓

2. VIEW MODE TOGGLE
   User clicks: "👔 Outfit Builder"
   ↓

3. SLOT OVERVIEW
   User sees: 4 collapsed slots with previews
   - Upper Body (12 items)
   - Lower Body (8 items)
   - Accessories (6 items)
   - Footwear (10 items)
   ↓

4. EXPAND SLOT
   User clicks: "👕 Upper Body"
   ↓

5. CAROUSEL INTERACTION
   User can:
   - Drag left/right to scroll
   - Click arrow buttons
   - View cards with reasoning
   ↓

6. PRODUCT SELECTION
   User clicks card
   ↓

7. PRODUCT DETAILS
   Modal shows full product info
   (Future: Add to outfit builder)
```

## Card Layout

```
┌────────────────────────────┐
│  #3          89%           │ ← Rank & Score badges
├────────────────────────────┤
│                            │
│                            │
│      Product Image         │
│      (300x360px)           │
│                            │
│                            │
├────────────────────────────┤
│  Summer Tank Top           │ ← Product name
├────────────────────────────┤
│  Comfortable cotton tank   │ ← Description
│  for warm days...          │
├────────────────────────────┤
│  💡 Why this?              │
│  Great match for your      │ ← AI Reasoning
│  style, in pink, as a tank │
├────────────────────────────┤
│  🎨 Pink  📦 Tank Top      │ ← Metadata tags
└────────────────────────────┘

[On Hover: Semi-transparent overlay]
┌────────────────────────────┐
│                            │
│     [View Details]         │ ← CTA Button
│                            │
└────────────────────────────┘
```

## Color Scheme

```
Primary Gradient:
#667eea ──────────► #764ba2
(Blue Purple)    (Deep Purple)

Backgrounds:
White:    #ffffff
Light:    #fafafa
Subtle:   #f8f9ff

Text Colors:
Dark:     #222222
Medium:   #666666
Light:    #888888

Accents:
Success:  #10b981
Warning:  #f59e0b
Error:    #ef4444
```

## Responsive Breakpoints

```
Desktop (>1200px):
┌─────────────────────────────────────┐
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│  │    │ │    │ │    │ │    │  ...  │ → Show 3-4 cards
│  └────┘ └────┘ └────┘ └────┘       │
└─────────────────────────────────────┘

Tablet (768-1200px):
┌───────────────────────────┐
│  ┌────┐ ┌────┐ ┌────┐    │
│  │    │ │    │ │    │ ...│ → Show 2-3 cards
│  └────┘ └────┘ └────┘    │
└───────────────────────────┘

Mobile (<768px):
┌──────────────┐
│  ┌────┐      │
│  │    │  ... │ → Show 1-2 cards, swipe gestures
│  └────┘      │
└──────────────┘
```

## States

### Loading State
```
┌─────────────────────────────┐
│        ⟳ Spinner            │
│  Building your personalized │
│  outfit...                  │
└─────────────────────────────┘
```

### Empty State
```
┌─────────────────────────────┐
│  No recommendations found.  │
│  Try a different search.    │
└─────────────────────────────┘
```

### Error State
```
┌─────────────────────────────┐
│  ⚠️ Search failed.          │
│  Make sure the BFF server   │
│  is running.                │
└─────────────────────────────┘
```

## Animation Timeline

```
0ms:   User clicks "Outfit Builder"
       └─► Loading spinner appears

200ms: API request sent
       └─► Loading state shown

500ms: Response received
       └─► Slots start appearing (staggered)

600ms: Slot 1 fades in
       └─► slideDown animation

700ms: Slot 2 fades in
       └─► slideDown animation

800ms: Slot 3 fades in
       └─► slideDown animation

...

User clicks slot:
0ms:   Slot expands
       └─► Height transition (300ms)

100ms: Reasoning appears
       └─► Fade in

200ms: Carousel appears
       └─► slideDown animation

300ms: Animation complete
       └─► User can interact
```

This visual guide should help you understand how the Outfit Builder UI works!
