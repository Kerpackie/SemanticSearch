using SemanticSearch.Nexus.Clients.Glyph;
using SemanticSearch.Nexus.Clients.Clip;
using SemanticSearch.Nexus.Clients.Mneme;
using SemanticSearch.Nexus.Clients.Arbiter;
using SemanticSearch.Nexus.Services;
using Microsoft.AspNetCore.Server.Kestrel.Core;

// Enable HTTP/2 over plain HTTP (required for gRPC without TLS)
AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

var builder = WebApplication.CreateBuilder(args);

// Configure Kestrel to support HTTP/2 over plain HTTP (required for gRPC without TLS)
builder.WebHost.ConfigureKestrel(options =>
{
    // Allow HTTP/2 over plain HTTP for all endpoints
    options.ConfigureEndpointDefaults(listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http2;
    });
    
    // Extended timeouts for long-running ML model operations
    options.Limits.KeepAliveTimeout = TimeSpan.FromMinutes(5);
    options.Limits.RequestHeadersTimeout = TimeSpan.FromMinutes(5);
});

// Add Aspire service defaults (service discovery, health checks, telemetry)
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddGrpc();
builder.Services.AddGrpcHealthChecks();

// Configure HTTP client for GptApi (NLP service)
var gptApiUrl = builder.Configuration["services:gpt-api:http:0"] 
                ?? builder.Configuration["services:gpt-api:https:0"]
                ?? "http://localhost:5107";
#pragma warning disable EXTEXP0001 // Experimental API - acceptable for PoC
builder.Services.AddHttpClient("GptApi", client =>
{
    client.BaseAddress = new Uri(gptApiUrl);
})
.ConfigureHttpClient(client => client.Timeout = TimeSpan.FromMinutes(5))
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    // Extended timeouts for LLM inference
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);
    // Circuit breaker sampling duration must be >= 2x attempt timeout
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(7);
});

// Configure HTTP client for Recommender service
var recommenderUrl = builder.Configuration["services:recommender:http:0"] 
                     ?? builder.Configuration["services:recommender:https:0"]
                     ?? "http://localhost:8000";
builder.Services.AddHttpClient("Recommender", client =>
{
    client.BaseAddress = new Uri(recommenderUrl);
})
.ConfigureHttpClient(client => client.Timeout = TimeSpan.FromMinutes(5))
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    // Extended timeouts for ML recommendations
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(2);
    // Circuit breaker sampling duration must be >= 2x attempt timeout
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(5);
});

// Configure gRPC clients for downstream services
// Glyph - Text embedding service
var glyphUrl = builder.Configuration["services:glyph:http:0"] 
               ?? builder.Configuration["services:glyph:https:0"]
               ?? "http://localhost:50051";
builder.Services.AddGrpcClient<Embedder.EmbedderClient>(options =>
{
    options.Address = new Uri(glyphUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true,
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
    KeepAlivePingDelay = TimeSpan.FromSeconds(60),
    KeepAlivePingTimeout = TimeSpan.FromSeconds(30)
})
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(7);
});

// Eidolon - Image embedding service (CLIP)
var eidolonUrl = builder.Configuration["services:eidolon:http:0"] 
                 ?? builder.Configuration["services:eidolon:https:0"]
                 ?? "http://localhost:50052";
builder.Services.AddGrpcClient<ClipEmbedder.ClipEmbedderClient>(options =>
{
    options.Address = new Uri(eidolonUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true,
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
    KeepAlivePingDelay = TimeSpan.FromSeconds(60),
    KeepAlivePingTimeout = TimeSpan.FromSeconds(30)
})
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(7);
});

// Mneme - Vector database service
var mnemeUrl = builder.Configuration["services:mneme-api:http:0"] 
               ?? builder.Configuration["services:mneme-api:https:0"]
               ?? "http://localhost:5108";
builder.Services.AddGrpcClient<ProductSearch.ProductSearchClient>(options =>
{
    options.Address = new Uri(mnemeUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true,
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
    KeepAlivePingDelay = TimeSpan.FromSeconds(60),
    KeepAlivePingTimeout = TimeSpan.FromSeconds(30)
})
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(7);
});

// Arbiter - Reranking service
var arbiterUrl = builder.Configuration["services:arbiter:http:0"] 
                 ?? builder.Configuration["services:arbiter:https:0"]
                 ?? "http://localhost:50053";
builder.Services.AddGrpcClient<Reranker.RerankerClient>(options =>
{
    options.Address = new Uri(arbiterUrl);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    EnableMultipleHttp2Connections = true,
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
    KeepAlivePingDelay = TimeSpan.FromSeconds(60),
    KeepAlivePingTimeout = TimeSpan.FromSeconds(30)
})
.RemoveAllResilienceHandlers()
.AddStandardResilienceHandler(options =>
{
    options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(5);
    options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(3);
    options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(7);
});
#pragma warning restore EXTEXP0001

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