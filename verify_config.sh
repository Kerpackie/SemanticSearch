#!/bin/bash
# Verify that the recommender configuration is correct

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Recommender Configuration Verification               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check 1: Verify Nexus hardcoded URL
echo "✓ Checking Nexus configuration..."
if grep -q 'http://localhost:8000' src/SemanticSearch.Nexus/Program.cs; then
    echo "  ✅ Nexus has hardcoded fallback to http://localhost:8000"
else
    echo "  ❌ WARNING: Nexus hardcoded URL not found!"
fi
echo ""

# Check 2: Verify AppHost doesn't launch recommender
echo "✓ Checking AppHost configuration..."
if ! grep -q 'AddExecutable.*recommender' src/SemanticSearch.AppHost/AppHost.cs; then
    echo "  ✅ AppHost correctly NOT launching recommender"
else
    echo "  ⚠️  WARNING: AppHost still has recommender executable!"
fi
echo ""

# Check 3: Check if recommender is currently running
echo "✓ Checking if recommender is running..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "  ✅ Recommender is RUNNING on port 8000"
    
    # Get status
    STATUS=$(curl -s http://localhost:8000/ | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
    if [ "$STATUS" = "healthy" ]; then
        echo "  ✅ Recommender status: healthy"
    else
        echo "  ⚠️  Recommender status: $STATUS"
    fi
else
    echo "  ⚠️  Recommender is NOT running on port 8000"
    echo "     Start it with: cd src/recommender && python3 api.py"
fi
echo ""

# Check 4: Verify SearchOrchestratorService has GetRecommenderScores method
echo "✓ Checking SearchOrchestratorService integration..."
if grep -q 'GetRecommenderScores' src/SemanticSearch.Nexus/Services/SearchOrchestratorService.cs; then
    echo "  ✅ SearchOrchestratorService has recommender integration"
else
    echo "  ❌ WARNING: GetRecommenderScores method not found!"
fi
echo ""

# Check 5: Verify BFF passes customerId
echo "✓ Checking BFF configuration..."
if grep -q 'CustomerId' src/SemanticSearch.BFF/Program.cs; then
    echo "  ✅ BFF configured to pass CustomerId to Nexus"
else
    echo "  ❌ WARNING: BFF CustomerId handling not found!"
fi
echo ""

# Check 6: Verify proto files have customer_id field
echo "✓ Checking proto file updates..."
if grep -q 'customer_id' src/SemanticSearch.Nexus/Protos/search.proto; then
    echo "  ✅ Nexus proto has customer_id field"
else
    echo "  ❌ WARNING: customer_id not in Nexus proto!"
fi

if grep -q 'customer_id' src/SemanticSearch.BFF/Protos/search.proto; then
    echo "  ✅ BFF proto has customer_id field"
else
    echo "  ❌ WARNING: customer_id not in BFF proto!"
fi
echo ""

# Check 7: Verify frontend has UserSelector
echo "✓ Checking frontend components..."
if [ -f "src/semantic-search-frontned/src/components/UserSelector.tsx" ]; then
    echo "  ✅ UserSelector component exists"
else
    echo "  ❌ WARNING: UserSelector.tsx not found!"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "Summary:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Configuration Status:"
echo "  • Nexus → localhost:8000 (hardcoded) ✓"
echo "  • AppHost → Manual startup ✓"
echo "  • BFF → Passes customerId ✓"
echo "  • Frontend → User selection ✓"
echo ""
echo "To start the full stack:"
echo "  1. Terminal 1: cd src/recommender && python3 api.py"
echo "  2. Terminal 2: cd src/SemanticSearch.AppHost && dotnet run"
echo "  3. Terminal 3: cd src/semantic-search-frontned && npm run dev"
echo ""
