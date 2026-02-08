# 🚀 Quick Reference Card

## Starting the Application

### Option 1: Full Stack via Aspire (Recommended)
```bash
cd src/SemanticSearch.AppHost
dotnet run
```
This starts:
- BFF
- Nexus
- Arbiter (Rust)
- Glyph (Rust)
- Eidolon (Rust)
- Recommender (Python)
- All infrastructure (Postgres, Qdrant, Redis)

### Option 2: Frontend Only (for UI development)
```bash
cd src/semantic-search-frontned
npm run dev
```
Opens at: `http://localhost:5173`

---

## Test Users (Hardcoded)

| Avatar | Name | CustomerId | Use Case |
|--------|------|------------|----------|
| 👤 | Guest | *(none)* | No personalization |
| 👩 | Emma Johnson | `00007d2de...` | Test persona 1 |
| 👨 | Michael Chen | `0000ca64f...` | Test persona 2 |
| 👱‍♀️ | Sarah Williams | `0001f1ccb...` | Test persona 3 |

---

## How to Test

1. **Open app** → Defaults to Guest mode
2. **Search** → e.g., "summer dress"
3. **Check Network tab** → No customerId in request
4. **Select Emma** → Personalization badge appears
5. **Search again** → customerId in request, different results
6. **Switch to Michael** → Results update automatically

---

## Key Files Modified

### Backend
- `AppHost.cs` - Added recommender service
- `SearchOrchestratorService.cs` - Hybrid scoring logic
- `search.proto` - Added customer_id field

### Frontend
- `UserSelector.tsx` - New component ⭐
- `App.tsx` - User state + integration
- `products.ts` - API with customerId

---

## Verification Checklist

- [ ] Recommender health: `curl http://localhost:8000/`
- [ ] User selector visible in UI
- [ ] Guest mode: No customerId sent
- [ ] Logged in: CustomerId sent
- [ ] Badge appears/disappears
- [ ] Results differ between users

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Recommender won't start | `pip3 install -r src/recommender/requirements_api.txt` |
| UI not updating | Hard refresh (Cmd+Shift+R) |
| CustomerId not sent | Select user (not Guest) |
| Proto errors | `dotnet clean && dotnet build` Nexus & BFF |

---

## Architecture at a Glance

```
┌─────────────┐
│  Frontend   │ User selects persona
│   (React)   │ → sends customerId
└──────┬──────┘
       │
┌──────▼──────┐
│     BFF     │ Forwards to Nexus
└──────┬──────┘
       │
┌──────▼──────┐
│    Nexus    │ if (customerId):
│             │   Arbiter (60%) +
│             │   Recommender (40%)
│             │ else:
│             │   Arbiter (100%)
└─────────────┘
```

---

## Documentation

- **RECOMMENDER_INTEGRATION.md** - Backend technical details
- **RECOMMENDER_QUICKSTART.md** - How to run & verify
- **UI_UPGRADE_SUMMARY.md** - Frontend changes
- **UI_TESTING_GUIDE.md** - Step-by-step testing
- **COMPLETE_IMPLEMENTATION.md** - Full overview

---

## Configuration

### Score Weights (Nexus)
File: `SearchOrchestratorService.cs` ~line 237
```csharp
finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore);
```

### Test Users (Frontend)
File: `UserSelector.tsx` ~line 11
```typescript
export const TEST_USERS: User[] = [...]
```

---

## Next Actions

1. **Test the integration** → See UI_TESTING_GUIDE.md
2. **Monitor performance** → Watch Nexus logs
3. **Gather feedback** → Note which personas work best
4. **Iterate** → Adjust score weights if needed

---

**Status:** ✅ Ready to Test

**Last Updated:** February 3, 2026
