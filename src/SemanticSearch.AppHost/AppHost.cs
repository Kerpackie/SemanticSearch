var builder = DistributedApplication.CreateBuilder(args);

var qdrant = builder.AddQdrant("qdrant")
    .WithLifetime(ContainerLifetime.Persistent)
    .WithDataVolume(); // or .WithDataBindMount("C:\\Qdrant\\Data")

builder.AddProject<Projects.Mneme_Api>("mneme-api")
    .WithReference(qdrant)
    .WaitFor(qdrant);

var cache = builder.AddRedis("cache");


var apiService = builder.AddProject<Projects.SemanticSearch_ApiService>("apiservice")
    .WithHttpHealthCheck("/health");

builder.AddProject<Projects.SemanticSearch_Web>("webfrontend")
    .WithExternalHttpEndpoints()
    .WithHttpHealthCheck("/health")
    .WithReference(cache)
    .WaitFor(cache)
    .WithReference(apiService)
    .WaitFor(apiService);

builder.Build().Run();