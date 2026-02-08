#!/bin/bash

# Outfit Builder Quick Start Script
# This script helps you quickly test the new Outfit Builder feature

set -e

echo "🎨 Starting Outfit Builder Demo"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "SemanticSearch.sln" ]; then
    echo "❌ Error: Please run this script from the SemanticSearch root directory"
    exit 1
fi

# Step 1: Build and start backend services
echo "📦 Step 1: Starting backend services..."
echo ""

# Check if services are already running
if curl -s http://localhost:5105/health > /dev/null 2>&1; then
    echo "✅ Backend services already running"
else
    echo "Starting backend with Aspire..."
    echo "Please start the AppHost project separately in your IDE"
    echo "Or run: cd src/SemanticSearch.AppHost && dotnet run"
    echo ""
    read -p "Press Enter when backend services are running..."
fi

# Step 2: Check if recommender is running
echo ""
echo "🤖 Step 2: Checking recommender service..."
if curl -s http://localhost:5002/health > /dev/null 2>&1; then
    echo "✅ Recommender service is running"
else
    echo "⚠️  Recommender service not running (optional for testing)"
    echo "To start it: cd src/recommender && python3 api.py"
fi

# Step 3: Start frontend
echo ""
echo "🌐 Step 3: Starting frontend..."
cd src/semantic-search-frontned

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the dev server
echo ""
echo "🚀 Starting Vite dev server..."
echo ""
echo "================================"
echo "✨ Outfit Builder is ready!"
echo "================================"
echo ""
echo "📍 Open: http://localhost:5173"
echo ""
echo "How to test:"
echo "1. Enter a search query (e.g., 'summer casual outfit')"
echo "2. Click the '👔 Outfit Builder' button"
echo "3. Explore the categorized slots"
echo "4. Click on slots to expand and see carousels"
echo "5. Scroll through top 10 items per slot"
echo ""
echo "Toggle between Grid View and Outfit Builder to compare!"
echo ""

npm run dev
