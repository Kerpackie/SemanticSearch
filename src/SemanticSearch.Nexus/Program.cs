using SemanticSearch.Nexus.Clients.Glyph;
using SemanticSearch.Nexus.Clients.Clip;
using SemanticSearch.Nexus.Clients.Mneme;
using SemanticSearch.Nexus.Clients.Arbiter;
using SemanticSearch.Nexus.Services;

var builder = WebApplication.CreateBuilder(args);

// Add Aspire service defaults (service discovery, health checks, telemetry)
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddGrpc();
builder.Services.AddGrpcHealthChecks();

// Configure HTTP client for GptApi (NLP service)
builder.Services.AddHttpClient("GptApi", client =>
{
    client.BaseAddress = new Uri("https+http://gpt-api");
});

// Configure gRPC clients for downstream services
// Glyph - Text embedding service
builder.Services.AddGrpcClient<Embedder.EmbedderClient>(options =>
{
    options.Address = new Uri("https+http://glyph");
});

// Eidolon - Image embedding service (CLIP)
builder.Services.AddGrpcClient<ClipEmbedder.ClipEmbedderClient>(options =>
{
    options.Address = new Uri("https+http://eidolon");
});

// Mneme - Vector database service
builder.Services.AddGrpcClient<ProductSearch.ProductSearchClient>(options =>
{
    options.Address = new Uri("https+http://mneme-api");
});

// Arbiter - Reranking service
builder.Services.AddGrpcClient<Reranker.RerankerClient>(options =>
{
    options.Address = new Uri("https+http://arbiter");
});

var app = builder.Build();

// Map Aspire default endpoints (health checks)
app.MapDefaultEndpoints();

// Configure the HTTP request pipeline.
app.MapGrpcService<GreeterService>();
app.MapGrpcService<SearchOrchestratorService>();
app.MapGrpcHealthChecksService();

app.MapGet("/",
    () =>
        "Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");

app.Run();