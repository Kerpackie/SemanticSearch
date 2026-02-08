# UI Upgrade Testing Guide

## Quick Start

1. **Navigate to frontend directory**:
   ```bash
   cd /Users/kerpackie/RiderProjects/SemanticSearch/src/semantic-search-frontned
   ```

2. **Install dependencies** (if not already done):
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   - The app should automatically open at `http://localhost:5173`
   - If not, manually navigate to that URL

## What to Test

### 1. User Selector Visibility
✅ **Check:** User selector appears in top-right of hero section  
✅ **Check:** Shows "Guest (No Personalization)" by default  
✅ **Check:** Clicking opens dropdown with 4 users  

### 2. User Selection
✅ **Test:** Click on "Emma Johnson"  
   - Dropdown should close
   - Button should now show "👩 Emma Johnson"
   - Personalization badge should appear: "✨ Personalized for Emma Johnson"

✅ **Test:** Click dropdown again and select different user  
   - Should switch smoothly
   - Badge text should update

✅ **Test:** Select "Guest (No Personalization)"  
   - Personalization badge should disappear
   - Button shows guest icon

### 3. Search Without Personalization (Guest Mode)

1. Make sure "Guest" is selected
2. Search for: **"summer dress"**
3. **Open Browser DevTools** (F12 or Cmd+Option+I)
4. Go to **Network** tab
5. Find the request to `/api/search`
6. Click on it and view **Request Payload**

✅ **Expected:** No `customerId` field in the request
```json
{
  "query": "summer dress",
  "limit": 50
}
```

### 4. Search With Personalization (Logged In)

1. Select "Emma Johnson" from dropdown
2. Notice personalization badge appears
3. Search for: **"summer dress"**
4. **Check Network tab** again for `/api/search` request

✅ **Expected:** `customerId` field should be present
```json
{
  "query": "summer dress",
  "limit": 50,
  "customerId": "00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f"
}
```

### 5. Dynamic User Switching

1. Search for: **"black jacket"**
2. Wait for results to load
3. **Without clearing search**, switch from Emma to Michael
4. Results should automatically reload
5. **Check Network tab** - should see new request with different `customerId`

✅ **Expected:** 
- New search triggered automatically
- Different customerId in request
- Potentially different ranking of results

### 6. Responsive Design

1. **Desktop view** (> 768px):
   - User selector on right side
   - Title and badge on left side
   - Horizontal layout

2. **Mobile view** (< 768px):
   - Resize browser window or use DevTools device mode
   - User selector should stack below title
   - Personalization badge should be on its own line
   - All elements should remain readable

### 7. Visual Polish

✅ **Check hover states:**
- User selector button has subtle hover effect
- Dropdown items highlight on hover
- Selected user has blue background

✅ **Check animations:**
- Dropdown appears smoothly
- Backdrop overlay fades in
- Click outside to close works

## Testing Scenarios

### Scenario 1: First-Time User
1. Open app (should default to Guest)
2. Browse products without searching
3. Try a search - should work with no personalization
4. Select a user - badge appears
5. Search again - results personalized

### Scenario 2: Power User
1. Select Emma Johnson
2. Search: "casual dress"
3. Switch to Michael Chen (without clearing search)
4. Observe different results
5. Switch to Sarah Williams
6. Compare all three result sets

### Scenario 3: Compare Guest vs Personalized
1. Search "winter jacket" as Guest
2. Note top 3 results
3. Select Emma Johnson
4. Search "winter jacket" again
5. Compare top 3 results - should be different if recommender is working

## Backend Verification

### Check Recommender Service is Running

```bash
curl http://localhost:8000/
```

✅ **Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/ffnn_.../model.keras",
  "metadata": {...}
}
```

### Check BFF is Running

```bash
curl http://localhost:XXXX/api/categories
```

✅ **Should return:** Array of category names

### Monitor Nexus Logs

When searching with a logged-in user, you should see:
```
Got recommender scores for X products
Product 123: Arbiter=0.85, Recommender=0.72, Final=0.79
```

## Common Issues

### Issue: User selector doesn't appear
**Solution:** 
- Check browser console for errors
- Verify all imports in App.tsx
- Check that UserSelector.css is loaded

### Issue: Dropdown doesn't close
**Solution:**
- Check that backdrop click handler is working
- Look for JavaScript errors in console
- Try clicking outside the dropdown

### Issue: No customerId in request
**Solution:**
- Verify user is selected (not Guest)
- Check `products.ts` API function
- Ensure `currentUser.id` is not empty string

### Issue: Results don't change between users
**Solution:**
- Verify recommender service is running
- Check Nexus logs for errors
- Ensure customer IDs match test data
- Check that article IDs in database are numeric

### Issue: Personalization badge doesn't appear
**Solution:**
- Check condition: `{currentUser?.id && ...}`
- Guest user has empty string `id: ''`
- Other users should have long hash IDs

## Success Criteria

✅ User can select from 4 different personas  
✅ Guest mode doesn't send customerId  
✅ Logged-in mode sends correct customerId  
✅ Personalization badge appears when logged in  
✅ Search results update when switching users  
✅ UI is responsive on mobile  
✅ Dropdown works smoothly  
✅ Backend receives customerId and combines scores  

## Next Steps After Testing

1. **Gather feedback**: Note which user personas produce best results
2. **Performance testing**: Measure latency difference between modes
3. **A/B testing**: Compare user satisfaction between modes
4. **Refinement**: Adjust Arbiter/Recommender weight ratios if needed
5. **Add more users**: Expand test persona list if needed

---

**Happy Testing!** 🎉

If you encounter issues, check:
1. Browser console for frontend errors
2. BFF logs for API errors  
3. Nexus logs for recommender integration errors
4. Recommender service health endpoint
