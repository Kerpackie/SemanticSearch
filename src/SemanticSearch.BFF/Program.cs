using SemanticSearch.BFF.Clients;

// Enable HTTP/2 over plain HTTP (required for gRPC without TLS)
AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

var builder = WebApplication.CreateBuilder(args);

// Add Aspire service defaults (service discovery, health checks, telemetry)
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

// Configure gRPC clients for downstream services
// Nexus - Search orchestration service
builder.Services.AddGrpcClient<SearchOrchestrator.SearchOrchestratorClient>(options =>
{
    var nexusUrl = builder.Configuration["services:nexus:http:0"] 
                   ?? builder.Configuration["services:nexus:https:0"]
                   ?? "http://localhost:5105";
    options.Address = new Uri(nexusUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true
});

// PgApi - Product service (PostgreSQL)
builder.Services.AddGrpcClient<ProductService.ProductServiceClient>(options =>
{
    var pgApiUrl = builder.Configuration["services:pg-api:http:0"] 
                   ?? builder.Configuration["services:pg-api:https:0"]
                   ?? "http://localhost:5106";
    options.Address = new Uri(pgApiUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.UseHttpsRedirection();

// Map Aspire default endpoints (health checks)
app.MapDefaultEndpoints();

// API Endpoints - now fetching from PostgreSQL via PgApi
app.MapGet("/api/products", async (string? category, string? search, int? page, int? pageSize, ProductService.ProductServiceClient pgApiClient) =>
{
    try
    {
        if (!string.IsNullOrEmpty(search))
        {
            // Use search endpoint
            var searchRequest = new SearchProductsRequest
            {
                Query = search,
                Limit = pageSize ?? 50
            };
            var response = await pgApiClient.SearchProductsAsync(searchRequest);
            return Results.Ok(MapProducts(response.Products, category));
        }
        else
        {
            // Use list endpoint with pagination
            var listRequest = new ListProductsRequest
            {
                Page = page ?? 1,
                PageSize = pageSize ?? 50
            };
            var response = await pgApiClient.ListProductsAsync(listRequest);
            return Results.Ok(new ProductListResult(MapProducts(response.Products, category), response.TotalCount));
        }
    }
    catch (Exception ex)
    {
        return Results.Problem($"Failed to fetch products: {ex.Message}");
    }
}).WithName("GetProducts");

app.MapGet("/api/products/{id}", async (string id, ProductService.ProductServiceClient pgApiClient) =>
{
    try
    {
        var request = new GetProductRequest { ArticleId = id };
        var response = await pgApiClient.GetProductAsync(request);
        return response.Product is null ? Results.NotFound() : Results.Ok(MapProduct(response.Product));
    }
    catch (Grpc.Core.RpcException ex) when (ex.StatusCode == Grpc.Core.StatusCode.NotFound)
    {
        return Results.NotFound();
    }
    catch (Exception ex)
    {
        return Results.Problem($"Failed to fetch product: {ex.Message}");
    }
}).WithName("GetProductById");

app.MapGet("/api/categories", () => new[] { "All", "Garment Upper body", "Garment Lower body", "Garment Full body", "Accessories", "Underwear", "Shoes", "Swimwear", "Unknown" }).WithName("GetCategories");

// Semantic Search endpoint - calls Nexus orchestrator
app.MapPost("/api/search", async (SearchRequest request, SearchOrchestrator.SearchOrchestratorClient nexusClient, ProductService.ProductServiceClient pgApiClient) =>
{
    try
    {
        // Call Nexus orchestrator for semantic search
        var grpcRequest = new TextSearchRequest
        {
            Query = request.Query,
            Limit = request.Limit ?? 20,
            EnableReranking = request.EnableReranking ?? true,
            CustomerId = request.CustomerId ?? ""
        };

        var grpcResponse = await nexusClient.SearchByTextAsync(grpcRequest);

        // Get full product details from PgApi for the search results
        var articleIds = grpcResponse.Results.Select(r => r.Id).ToList();
        
        List<Product> fullProducts = [];
        if (articleIds.Count > 0)
        {
            var productsRequest = new GetProductsRequest();
            productsRequest.ArticleIds.AddRange(articleIds);
            var productsResponse = await pgApiClient.GetProductsAsync(productsRequest);
            fullProducts = productsResponse.Products.ToList();
        }

        // Map results with full product data
        var searchResults = grpcResponse.Results.Select(r =>
        {
            var fullProduct = fullProducts.FirstOrDefault(p => p.ArticleId == r.Id);
            return new ProductSearchResult(
                r.Id,
                fullProduct?.ProdName ?? r.Name,
                fullProduct?.DetailDesc ?? r.Description,
                r.Score,
                r.Rank,
                fullProduct?.ProductGroupName ?? "Unknown",
                fullProduct?.ColourGroupName ?? "",
                fullProduct?.ProductTypeName ?? ""
            );
        }).ToList();

        return Results.Ok(new SemanticSearchResponse(
            searchResults,
            grpcResponse.ProcessedQuery,
            grpcResponse.TotalResults
        ));
    }
    catch (Exception ex)
    {
        // Fallback to PgApi search if Nexus is unavailable
        try
        {
            var searchRequest = new SearchProductsRequest
            {
                Query = request.Query,
                Limit = request.Limit ?? 20
            };
            var response = await pgApiClient.SearchProductsAsync(searchRequest);
            
            var fallbackResults = response.Products.Select((p, i) => new ProductSearchResult(
                p.ArticleId,
                p.ProdName,
                p.DetailDesc,
                1.0f - (i * 0.05f),
                i + 1,
                p.ProductGroupName,
                p.ColourGroupName,
                p.ProductTypeName
            )).ToList();

            return Results.Ok(new SemanticSearchResponse(fallbackResults, request.Query, fallbackResults.Count));
        }
        catch
        {
            return Results.Problem($"Search failed: {ex.Message}");
        }
    }
}).WithName("SemanticSearch");

// Outfit Search endpoint - categorizes results into slots and re-ranks top 10 per slot
app.MapPost("/api/outfit-search", async (OutfitSearchRequest request, SearchOrchestrator.SearchOrchestratorClient nexusClient, ProductService.ProductServiceClient pgApiClient, ILogger<Program> logger) =>
{
    try
    {
        logger.LogInformation("=== OUTFIT SEARCH STARTED ===");
        logger.LogInformation("Query: {Query}, CustomerId: {CustomerId}", request.Query, request.CustomerId);
        
        // Call Nexus orchestrator for semantic search with more results
        var grpcRequest = new TextSearchRequest
        {
            Query = request.Query,
            Limit = 100, // Get more results to categorize
            EnableReranking = true,
            CustomerId = request.CustomerId ?? ""
        };

        logger.LogInformation("Calling Nexus with limit: 100, EnableReranking: TRUE");
        var grpcResponse = await nexusClient.SearchByTextAsync(grpcRequest);
        logger.LogInformation("Nexus returned {Count} results, TotalResults: {Total}", 
            grpcResponse.Results.Count, grpcResponse.TotalResults);

        // Log first few IDs and SCORES to see if reranking actually happened
        if (grpcResponse.Results.Count > 0)
        {
            logger.LogInformation("Sample results from Nexus (first 5 with SCORES):");
            foreach (var result in grpcResponse.Results.Take(5))
            {
                logger.LogInformation("  - ID: {Id}, Name: {Name}, SCORE: {Score:F4}", 
                    result.Id, result.Name, result.Score);
            }
            
            // Check score distribution
            var avgScore = grpcResponse.Results.Average(r => r.Score);
            var maxScore = grpcResponse.Results.Max(r => r.Score);
            var minScore = grpcResponse.Results.Min(r => r.Score);
            logger.LogInformation("Score stats - Avg: {Avg:F4}, Max: {Max:F4}, Min: {Min:F4}", 
                avgScore, maxScore, minScore);
                
            if (maxScore < 0.6f)
            {
                logger.LogWarning("⚠️ WARNING: All scores below 0.6! Reranking might have FAILED!");
                logger.LogWarning("⚠️ This looks like raw vector similarity, not cross-encoder scores!");
            }
        }

        // Get full product details from PgApi
        // Note: r.Id is the document UUID from vector DB, but we need the actual article_id from metadata
        var articleIds = grpcResponse.Results
            .Select(r => {
                // Try to get article_id from metadata first
                if (r.Metadata.TryGetValue("article_id", out var articleId) && !string.IsNullOrEmpty(articleId))
                {
                    return articleId;
                }
                // Fallback to using the ID directly (in case it's already the article ID)
                return r.Id;
            })
            .Where(id => !string.IsNullOrEmpty(id))
            .ToList();
            
        logger.LogInformation("Article IDs to fetch: {Count}", articleIds.Count);
        
        List<Product> fullProducts = [];
        if (articleIds.Count > 0)
        {
            var productsRequest = new GetProductsRequest();
            productsRequest.ArticleIds.AddRange(articleIds);
            logger.LogInformation("Fetching product details from PgApi for {Count} articles", articleIds.Count);
            logger.LogInformation("Sample article IDs being sent to PgApi: {Ids}", 
                string.Join(", ", articleIds.Take(3)));
            
            var productsResponse = await pgApiClient.GetProductsAsync(productsRequest);
            fullProducts = productsResponse.Products.ToList();
            logger.LogInformation("PgApi returned {Count} full products", fullProducts.Count);
            
            if (fullProducts.Count == 0)
            {
                logger.LogWarning("⚠️ PgApi returned ZERO products! ID format mismatch?");
            }
            else if (fullProducts.Count < articleIds.Count)
            {
                logger.LogWarning("⚠️ PgApi returned only {Returned} of {Requested} products", 
                    fullProducts.Count, articleIds.Count);
            }
        }

        // Categorize products into slots based on product group
        logger.LogInformation("Starting categorization into slots...");
        var slots = CategorizeIntoSlots(grpcResponse.Results, fullProducts, articleIds);
        logger.LogInformation("Categorization complete. Slot counts:");
        foreach (var (slotName, slotData) in slots)
        {
            logger.LogInformation("  - {SlotName}: {Count} items", slotName, slotData.Recommendations.Count);
        }

        var response = new OutfitSearchResponse(
            slots,
            grpcResponse.TotalResults,
            grpcResponse.ProcessedQuery
        );
        
        logger.LogInformation("=== OUTFIT SEARCH COMPLETED === Total slots: {SlotCount}", slots.Count);
        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Outfit search failed: {Message}", ex.Message);
        return Results.Problem($"Outfit search failed: {ex.Message}");
    }
}).WithName("OutfitSearch");

// Helper methods (must be before app.Run())
static Dictionary<string, SlotData> CategorizeIntoSlots(
    IEnumerable<SearchResult> results, 
    List<Product> fullProducts,
    List<string> articleIds)
{
    Console.WriteLine($"[CategorizeIntoSlots] Starting with {results.Count()} results and {fullProducts.Count} products");
    
    var slots = new Dictionary<string, SlotData>
    {
        ["upper_body"] = new SlotData("upper_body", [], "Items for your upper body like shirts, tops, jackets, and sweaters"),
        ["lower_body"] = new SlotData("lower_body", [], "Items for your lower body like pants, jeans, skirts, and shorts"),
        ["full_body"] = new SlotData("full_body", [], "Complete outfits like dresses and jumpsuits"),
        ["shoes"] = new SlotData("shoes", [], "Footwear to complete your look"),
        ["accessories"] = new SlotData("accessories", [], "Accessories to enhance your style"),
        ["underwear"] = new SlotData("underwear", [], "Comfortable and stylish undergarments"),
        ["swimwear"] = new SlotData("swimwear", [], "Swimwear for beach and pool")
    };

    int processedCount = 0;
    int categorizedCount = 0;
    
    var resultsArray = results.ToArray();
    
    for (int i = 0; i < resultsArray.Length; i++)
    {
        var result = resultsArray[i];
        var articleId = i < articleIds.Count ? articleIds[i] : result.Id;
        
        processedCount++;
        var fullProduct = fullProducts.FirstOrDefault(p => p.ArticleId == articleId);
        
        if (fullProduct == null)
        {
            if (processedCount <= 5)
            {
                Console.WriteLine($"[CategorizeIntoSlots] WARNING: No full product found for articleId: {articleId} (from result ID: {result.Id})");
            }
            continue;
        }
        
        var productGroup = fullProduct?.ProductGroupName?.ToLowerInvariant() ?? "";
        var productType = fullProduct?.ProductTypeName?.ToLowerInvariant() ?? "";
        var garmentGroup = fullProduct?.GarmentGroupName?.ToLowerInvariant() ?? "";

        if (processedCount <= 5) // Log first 5 products
        {
            Console.WriteLine($"[CategorizeIntoSlots] Product {processedCount}: ID={result.Id}");
            Console.WriteLine($"  - Name: {fullProduct?.ProdName}");
            Console.WriteLine($"  - ProductGroup: '{productGroup}'");
            Console.WriteLine($"  - ProductType: '{productType}'");
            Console.WriteLine($"  - GarmentGroup: '{garmentGroup}'");
        }

        var recommendation = new RecommendationDto(
            articleId, // Use the actual article ID, not the document UUID
            fullProduct?.ProdName ?? result.Name,
            fullProduct?.DetailDesc ?? result.Description,
            result.Score,
            GenerateReasoning(result, fullProduct),
            new RecommendationMetadata(
                fullProduct?.ColourGroupName ?? "",
                fullProduct?.ProductTypeName ?? "",
                fullProduct?.ProductGroupName ?? ""
            )
        );

        bool categorized = false;
        
        // Categorize based on product group, type, and garment group
        // Upper Body
        if (productGroup.Contains("upper") || 
            garmentGroup.Contains("upper") ||
            garmentGroup.Contains("jersey") ||
            garmentGroup.Contains("knitwear") ||
            productType.Contains("shirt") || 
            productType.Contains("top") || 
            productType.Contains("t-shirt") ||
            productType.Contains("tee") ||
            productType.Contains("blouse") ||
            productType.Contains("sweater") ||
            productType.Contains("cardigan") ||
            productType.Contains("jacket") ||
            productType.Contains("coat") ||
            productType.Contains("vest") ||
            productType.Contains("hoodie"))
        {
            slots["upper_body"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: UPPER_BODY");
        }
        // Lower Body
        else if (productGroup.Contains("lower") || 
                 garmentGroup.Contains("lower") ||
                 garmentGroup.Contains("trousers") ||
                 productType.Contains("trouser") || 
                 productType.Contains("pants") ||
                 productType.Contains("jeans") || 
                 productType.Contains("skirt") || 
                 productType.Contains("shorts") ||
                 productType.Contains("leggings") ||
                 productType.Contains("chinos"))
        {
            slots["lower_body"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: LOWER_BODY");
        }
        // Full Body
        else if (productGroup.Contains("full") || 
                 garmentGroup.Contains("dress") ||
                 productType.Contains("dress") || 
                 productType.Contains("jumpsuit") ||
                 productType.Contains("playsuit") ||
                 productType.Contains("romper") ||
                 productType.Contains("dungarees") ||
                 productType.Contains("overall"))
        {
            slots["full_body"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: FULL_BODY");
        }
        // Shoes
        else if (productGroup.Contains("shoes") || 
                 garmentGroup.Contains("shoes") ||
                 productType.Contains("shoe") || 
                 productType.Contains("boot") || 
                 productType.Contains("sneaker") ||
                 productType.Contains("trainer") ||
                 productType.Contains("sandal") ||
                 productType.Contains("slipper") ||
                 productType.Contains("pump") ||
                 productType.Contains("heel"))
        {
            slots["shoes"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: SHOES");
        }
        // Accessories
        else if (productGroup.Contains("accessories") || 
                 garmentGroup.Contains("accessories") ||
                 productType.Contains("bag") || 
                 productType.Contains("belt") || 
                 productType.Contains("hat") ||
                 productType.Contains("cap") ||
                 productType.Contains("scarf") ||
                 productType.Contains("gloves") ||
                 productType.Contains("jewelry") ||
                 productType.Contains("jewellery") ||
                 productType.Contains("necklace") ||
                 productType.Contains("bracelet") ||
                 productType.Contains("earring") ||
                 productType.Contains("watch") ||
                 productType.Contains("sunglasses") ||
                 productType.Contains("tie"))
        {
            slots["accessories"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: ACCESSORIES");
        }
        // Underwear
        else if (productGroup.Contains("underwear") || 
                 garmentGroup.Contains("underwear") ||
                 garmentGroup.Contains("socks") ||
                 productType.Contains("underwear") || 
                 productType.Contains("bra") ||
                 productType.Contains("brief") ||
                 productType.Contains("boxer") ||
                 productType.Contains("sock") ||
                 productType.Contains("tights"))
        {
            slots["underwear"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: UNDERWEAR");
        }
        // Swimwear
        else if (productGroup.Contains("swimwear") || 
                 garmentGroup.Contains("swimwear") ||
                 productType.Contains("swimwear") || 
                 productType.Contains("bikini") ||
                 productType.Contains("swimsuit") ||
                 productType.Contains("trunks") ||
                 productType.Contains("beachwear"))
        {
            slots["swimwear"].Recommendations.Add(recommendation);
            categorized = true;
            if (processedCount <= 5) Console.WriteLine($"  -> Categorized as: SWIMWEAR");
        }
        
        if (categorized)
        {
            categorizedCount++;
        }
        else if (processedCount <= 5)
        {
            Console.WriteLine($"  -> NOT CATEGORIZED!");
        }
    }

    Console.WriteLine($"[CategorizeIntoSlots] Processed {processedCount} products, categorized {categorizedCount}");
    Console.WriteLine($"[CategorizeIntoSlots] Slot breakdown BEFORE filtering:");
    foreach (var (slotName, slotData) in slots)
    {
        Console.WriteLine($"  - {slotName}: {slotData.Recommendations.Count} items");
        if (slotData.Recommendations.Count > 0)
        {
            Console.WriteLine($"    First item: {slotData.Recommendations[0].Name} (Score: {slotData.Recommendations[0].Score})");
            if (slotData.Recommendations.Count > 1)
            {
                Console.WriteLine($"    Last item: {slotData.Recommendations[slotData.Recommendations.Count - 1].Name} (Score: {slotData.Recommendations[slotData.Recommendations.Count - 1].Score})");
            }
        }
    }

    // Keep only top 10 for each slot and remove empty slots
    var filteredSlots = new Dictionary<string, SlotData>();
    foreach (var (key, slot) in slots)
    {
        if (slot.Recommendations.Count > 0)
        {
            var originalCount = slot.Recommendations.Count;
            slot.Recommendations = slot.Recommendations
                .OrderByDescending(r => r.Score)
                .Take(10)
                .ToList();
            filteredSlots[key] = slot;
            Console.WriteLine($"[CategorizeIntoSlots] {key}: kept {slot.Recommendations.Count} of {originalCount} items (top 10)");
        }
        else
        {
            Console.WriteLine($"[CategorizeIntoSlots] {key}: EMPTY - not included in response");
        }
    }

    Console.WriteLine($"[CategorizeIntoSlots] Returning {filteredSlots.Count} non-empty slots");
    return filteredSlots;
}

static string GenerateReasoning(SearchResult result, Product? fullProduct)
{
    var reasons = new List<string>();
    
    if (result.Score > 0.9f)
        reasons.Add("Perfect match for your search");
    else if (result.Score > 0.75f)
        reasons.Add("Great match for your style");
    else if (result.Score > 0.6f)
        reasons.Add("Good alternative option");
    
    if (fullProduct != null)
    {
        if (!string.IsNullOrEmpty(fullProduct.ColourGroupName))
            reasons.Add($"in {fullProduct.ColourGroupName.ToLower()}");
        
        if (!string.IsNullOrEmpty(fullProduct.ProductTypeName))
            reasons.Add($"as a {fullProduct.ProductTypeName.ToLower()}");
    }
    
    return reasons.Count > 0 ? string.Join(", ", reasons) : "Recommended for you";
}

static List<ProductDto> MapProducts(IEnumerable<Product> products, string? categoryFilter)
{
    var result = products.Select(MapProduct);
    
    if (!string.IsNullOrEmpty(categoryFilter) && categoryFilter != "All")
    {
        result = result.Where(p => p.ProductGroupName.Contains(categoryFilter, StringComparison.OrdinalIgnoreCase));
    }
    
    return result.ToList();
}

static ProductDto MapProduct(Product p) => new(
    p.ArticleId,
    p.ProductCode,
    p.ProdName,
    p.DetailDesc,
    p.ProductTypeName,
    p.ProductGroupName,
    p.ColourGroupName,
    p.PerceivedColourMasterName,
    p.GraphicalAppearanceName,
    p.DepartmentName,
    p.IndexName,
    p.IndexGroupName,
    p.SectionName,
    p.GarmentGroupName
);

app.Run();

// Records for the API
record ProductDto(
    string ArticleId,
    int ProductCode,
    string Name,
    string Description,
    string ProductType,
    string ProductGroupName,
    string ColourGroupName,
    string ColourMasterName,
    string GraphicalAppearance,
    string Department,
    string IndexName,
    string IndexGroupName,
    string Section,
    string GarmentGroup);

record ProductListResult(List<ProductDto> Products, int TotalCount);

record SearchRequest(string Query, int? Limit = 20, bool? EnableReranking = true, string? CustomerId = null);

record SemanticSearchResponse(List<ProductSearchResult> Products, string ProcessedQuery, int TotalResults);

record ProductSearchResult(
    string Id,
    string Name,
    string Description,
    float Score,
    int Rank,
    string ProductGroup,
    string Colour,
    string ProductType);

// Outfit Search DTOs
record OutfitSearchRequest(string Query, string? CustomerId = null);

record OutfitSearchResponse(
    Dictionary<string, SlotData> Slots,
    int TotalResults,
    string ProcessedQuery);

record SlotData(
    string SlotType,
    List<RecommendationDto> Recommendations,
    string? Reasoning)
{
    public List<RecommendationDto> Recommendations { get; set; } = Recommendations;
}

record RecommendationDto(
    string Id,
    string Name,
    string Description,
    float Score,
    string? Reasoning,
    RecommendationMetadata? Metadata);

record RecommendationMetadata(
    string Colour,
    string ProductType,
    string ProductGroup);

