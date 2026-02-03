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

app.Run();

// Helper methods
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
