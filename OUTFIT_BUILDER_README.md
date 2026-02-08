# Outfit Builder Feature

## Overview

The Outfit Builder is an innovative UI feature for the semantic search system that transforms search results into an RPG-style "outfit builder" interface. Instead of showing a flat grid of products, it categorizes results into different "slots" (like Upper Body, Lower Body, Shoes, Accessories) and presents each slot as a horizontal carousel with the top 10 re-ranked items.

## Features

### 🎨 Slot-Based Organization
Products are automatically categorized into intuitive slots:
- **Upper Body**: Shirts, tops, jackets, sweaters
- **Lower Body**: Pants, jeans, skirts, shorts
- **Full Body**: Dresses, jumpsuits, full outfits
- **Footwear**: Shoes, boots, sneakers, sandals
- **Accessories**: Bags, belts, hats, jewelry
- **Underwear**: Undergarments and intimate wear
- **Swimwear**: Swimsuits, bikinis, beachwear

### 🎯 Smart Re-ranking
- Each slot shows the top 10 items ranked by relevance score
- Items are re-ranked using the recommender system when a user is logged in
- Personalized scoring combines semantic similarity with user preferences

### 💡 Reasoning & Explanations
- Each slot includes reasoning for why it was created
- Individual recommendations show why they were selected
- Transparency in AI decision-making

### 🎠 Interactive Carousel UI
- Horizontal scrolling carousel for each slot
- Drag to scroll or use navigation arrows
- Mobile-friendly with touch gestures
- Smooth animations and transitions

### 🔄 View Mode Toggle
- Switch between traditional Grid View and Outfit Builder
- Seamless transition between modes
- Preference persists during session

## Architecture

### Frontend Components

```
OutfitBuilder.tsx          - Main container component
  ├── OutfitSlot.tsx        - Individual slot with expand/collapse
  │   └── SlotCarousel.tsx  - Horizontal carousel with navigation
  │       └── RecommendationCard.tsx - Individual product card with reasoning
```

### Backend Endpoint

**POST** `/api/outfit-search`

Request:
```json
{
  "query": "casual summer outfit",
  "customerId": "user123" // optional
}
```

Response:
```json
{
  "slots": {
    "upper_body": {
      "slotType": "upper_body",
      "reasoning": "Items for your upper body like shirts, tops, jackets, and sweaters",
      "recommendations": [
        {
          "id": "12345",
          "name": "Cotton T-Shirt",
          "description": "Comfortable cotton tee",
          "score": 0.95,
          "reasoning": "Perfect match for your search, in blue, as a t-shirt",
          "metadata": {
            "colour": "Blue",
            "productType": "T-shirt",
            "productGroup": "Garment Upper body"
          }
        }
        // ... up to 10 items
      ]
    }
    // ... other slots
  },
  "totalResults": 100,
  "processedQuery": "casual summer outfit"
}
```

### Categorization Logic

The backend uses product metadata to categorize items:

1. **Extract metadata**: Product group and type from database
2. **Apply rules**: String matching on product types and groups
3. **Add to slots**: Each product can belong to one slot
4. **Rank within slot**: Order by relevance score
5. **Top 10**: Keep only the best 10 items per slot
6. **Filter empty**: Remove slots with no items

### Reasoning Generation

Each recommendation includes AI-generated reasoning:
- Score-based quality ("Perfect match", "Great match", "Good alternative")
- Color information when available
- Product type context
- Personalization hints when logged in

## Usage

### For Users

1. **Search for products**: Enter a query like "summer beach outfit"
2. **Toggle Outfit Builder**: Click the "👔 Outfit Builder" button
3. **Explore slots**: Click on any slot to expand and see recommendations
4. **Browse carousel**: Scroll through the top 10 items in each slot
5. **View details**: Click on any product card to see full details
6. **See reasoning**: Each card shows why it was recommended

### For Developers

#### Enable Outfit Mode in Component

```typescript
import { OutfitBuilder } from './components/OutfitBuilder';
import { outfitSearch } from './api/products';

// Fetch outfit data
const response = await outfitSearch(query, customerId);

// Render outfit builder
<OutfitBuilder 
  slots={response.slots}
  loading={loading}
  onProductClick={(productId, slotType) => {
    // Handle product click
  }}
/>
```

#### Customize Slot Categories

Edit `Program.cs` in `SemanticSearch.BFF`:

```csharp
var slots = new Dictionary<string, SlotData>
{
    ["custom_slot"] = new SlotData(
        "custom_slot", 
        [], 
        "Your custom slot description"
    ),
    // ... other slots
};
```

#### Adjust Re-ranking Logic

Modify the categorization logic in `CategorizeIntoSlots()`:

```csharp
if (productType.Contains("your_custom_type"))
{
    slots["your_slot"].Recommendations.Add(recommendation);
}
```

## Styling

The outfit builder uses a cohesive design system:

- **Colors**: Purple gradient theme (#667eea to #764ba2)
- **Cards**: White background with subtle shadows
- **Animations**: Smooth transitions and hover effects
- **Responsive**: Mobile-first design with breakpoints

### Customize Appearance

Edit the CSS files in `src/components/`:
- `OutfitBuilder.css` - Main container styles
- `OutfitSlot.css` - Slot card and header styles
- `SlotCarousel.css` - Carousel navigation and scrolling
- `RecommendationCard.css` - Product card styling

## Performance Considerations

- **Lazy loading**: Slots only load carousel when expanded
- **Image optimization**: Images loaded on-demand with fallbacks
- **Efficient scrolling**: Native browser scroll with smooth behavior
- **Limited results**: Top 10 per slot reduces payload size

## Future Enhancements

- [ ] **Mix & Match**: Allow users to select one item from each slot to build complete outfits
- [ ] **Save Outfits**: Let users save their favorite combinations
- [ ] **Style Compatibility**: Check if items from different slots work well together
- [ ] **Virtual Try-On**: AR/VR integration for outfit visualization
- [ ] **Social Sharing**: Share outfit combinations with friends
- [ ] **Purchase Bundle**: Add all items from an outfit to cart at once
- [ ] **Alternative Items**: "Similar items" for each slot
- [ ] **Season Detection**: Automatically filter slots by season

## Troubleshooting

### Outfit Builder Not Showing

1. Verify the BFF server is running
2. Check browser console for API errors
3. Ensure `/api/outfit-search` endpoint is accessible
4. Verify search query returns results

### Empty Slots

- Check product categorization rules
- Verify product metadata in database
- Adjust matching criteria in `CategorizeIntoSlots()`

### Carousel Not Scrolling

- Check browser compatibility (modern browsers required)
- Verify CSS is loaded correctly
- Test with different viewport sizes

### Images Not Loading

- Verify images exist in `/public/images/` directory
- Check image naming convention (article ID + .jpg)
- Fallback placeholder should appear if images missing

## API Integration

The outfit search integrates with existing services:

- **Nexus**: Semantic search orchestration
- **Mneme**: Vector similarity search
- **Arbiter**: Re-ranking with LLM
- **Recommender**: Personalization scoring
- **PgApi**: Product database access

## Testing

```bash
# Start the development server
cd src/semantic-search-frontned
npm run dev

# Test the outfit search
# 1. Navigate to http://localhost:5173
# 2. Enter a search query
# 3. Click "Outfit Builder" toggle
# 4. Verify slots appear and are interactive
```

## License

Part of the SemanticSearch project.
