var builder = DistributedApplication.CreateBuilder(args);


var qdrant = builder.AddQdrant("qdrant")
    .WithLifetime(ContainerLifetime.Persistent)
    .WithDataVolume(); // or .WithDataBindMount("C:\\Qdrant\\Data")

var postgres = builder.AddPostgres("postgres")
    .WithLifetime(ContainerLifetime.Persistent)
    .WithPgWeb()
    .WithPgAdmin()
    .WithDataVolume();

var productDb = postgres.AddDatabase("HM", "products");

var pgApi = builder.AddProject<Projects.PgApi>("pg-api")
    .WithReference(postgres)
    .WaitFor(postgres)
    .WithHttpHealthCheck("/health");

var dataSeeder = builder.AddProject<Projects.DatabaseSeeder>("dataseeder")
    .WithReference(postgres)
    .WaitFor(postgres);

var mnemeApi = builder.AddProject<Projects.Mneme_Api>("mneme-api")
    .WithReference(qdrant)
    .WaitFor(qdrant);

builder.AddProject<Projects.MnemeTester>("mneme-tester")
    .WithReference(mnemeApi)
    .WaitFor(mnemeApi);

var cache = builder.AddRedis("cache");

var test = builder.AddProject<Projects.GptApi>("gpt-api")
    .WithHttpHealthCheck("/health");

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