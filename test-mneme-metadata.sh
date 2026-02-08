#!/bin/bash

echo "=== Mneme Metadata Test Runner ==="
echo ""
echo "This script helps you test if Mneme is correctly retrieving article_id metadata from Qdrant."
echo ""

# Check if AppHost is running
if ! pgrep -f "SemanticSearch.AppHost" > /dev/null; then
    echo "⚠️  WARNING: SemanticSearch.AppHost doesn't appear to be running"
    echo ""
    echo "Please start the AppHost first:"
    echo "  cd src/SemanticSearch.AppHost"
    echo "  dotnet run"
    echo ""
    read -p "Press Enter if AppHost is running, or Ctrl+C to exit..."
fi

# Try to find Mneme port from common locations
echo "Looking for Mneme service port..."
echo ""

MNEME_PORT=""

# Check if lsof is available and find ports
if command -v lsof &> /dev/null; then
    # Look for any dotnet process that might be mneme
    PORTS=$(lsof -ti:7000-8000 2>/dev/null | head -5)
    if [ -n "$PORTS" ]; then
        echo "Found services running on ports in range 7000-8000"
        echo "Check the Aspire dashboard to find the exact mneme-api port"
    fi
fi

echo ""
echo "📌 To find the Mneme port:"
echo "  1. Open the Aspire dashboard (usually http://localhost:15888)"
echo "  2. Look for 'mneme-api' service"
echo "  3. Note the HTTP/gRPC port number"
echo ""

read -p "Enter Mneme port number (or press Enter for default 5002): " PORT_INPUT

if [ -z "$PORT_INPUT" ]; then
    MNEME_PORT=5002
else
    MNEME_PORT=$PORT_INPUT
fi

echo ""
echo "Testing Mneme on port $MNEME_PORT..."
echo ""

cd "$(dirname "$0")"
dotnet run --project src/MnemeTest $MNEME_PORT

echo ""
echo "=== Test Complete ==="
