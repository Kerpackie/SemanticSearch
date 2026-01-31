var builder = DistributedApplication.CreateBuilder(args);

// =============================================================================
// Infrastructure
// =============================================================================

var qdrant = builder.AddQdrant("qdrant")
    .WithLifetime(ContainerLifetime.Persistent)
    .WithDataVolume();

var postgres = builder.AddPostgres("postgres")
    .WithLifetime(ContainerLifetime.Persistent)
    .WithPgWeb()
    .WithPgAdmin()
    .WithDataVolume();

var hmDatabase = postgres.AddDatabase("HM", "products");

var cache = builder.AddRedis("cache");

// =============================================================================
// Core Services (Rust gRPC microservices - running as native executables for Metal support)
// =============================================================================

// Glyph - Text embedding service (Rust)
var glyph = builder.AddExecutable("glyph", "cargo", "../Glyph", "run", "--release", "--bin", "Glyph")
    .WithHttpEndpoint(port: 50051, name: "grpc", env: "GRPC_PORT");

// Eidolon - Image embedding service / CLIP (Rust)
var eidolon = builder.AddExecutable("eidolon", "cargo", "../Eidolon", "run", "--release", "--bin", "Eidolon")
    .WithHttpEndpoint(port: 50052, name: "grpc", env: "GRPC_PORT");

// Arbiter - Reranking service (Rust)
var arbiter = builder.AddExecutable("arbiter", "cargo", "../arbiter", "run", "--release", "--bin", "arbiter")
    .WithHttpEndpoint(port: 50053, name: "grpc", env: "GRPC_PORT");

// =============================================================================
// .NET Services
// =============================================================================

var pgApi = builder.AddProject<Projects.PgApi>("pg-api")
    .WithReference(hmDatabase)
    .WaitFor(hmDatabase)
    .WithHttpHealthCheck("/health");

_ = builder.AddProject<Projects.DatabaseSeeder>("dataseeder")
    .WithReference(hmDatabase)
    .WaitFor(hmDatabase);

// Mneme - Vector database API
var mnemeApi = builder.AddProject<Projects.Mneme_Api>("mneme-api")
    .WithReference(qdrant)
    .WaitFor(qdrant);

builder.AddProject<Projects.MnemeTester>("mneme-tester")
    .WithReference(mnemeApi)
    .WaitFor(mnemeApi);

// GptApi - NLP/LLM service
var gptApi = builder.AddProject<Projects.GptApi>("gpt-api")
    .WithHttpHealthCheck("/health");

// =============================================================================
// Nexus - Orchestration Service
// Flow: BFF -> Nexus -> GptApi -> Glyph/Eidolon -> Mneme -> Arbiter -> BFF
// =============================================================================

var nexus = builder.AddProject<Projects.SemanticSearch_Nexus>("nexus")
    .WithReference(gptApi)
    .WithReference(mnemeApi)
    .WithReference(glyph.GetEndpoint("grpc"))
    .WithReference(eidolon.GetEndpoint("grpc"))
    .WithReference(arbiter.GetEndpoint("grpc"))
    .WaitFor(gptApi)
    .WaitFor(mnemeApi);

// =============================================================================
// BFF - Backend for Frontend
// =============================================================================

_ = builder.AddProject<Projects.SemanticSearch_BFF>("bff")
    .WithExternalHttpEndpoints()
    .WithReference(nexus)
    .WithReference(pgApi)
    .WaitFor(nexus)
    .WaitFor(pgApi)
    .WithHttpHealthCheck("/health");

// =============================================================================
// Frontend - React/Vite application (uses pnpm)
// =============================================================================

// Note: Frontend is a Vite/React app that runs independently
// Run `pnpm dev` in the frontend directory, or uncomment below when using Aspire.Hosting.NodeJs
// var frontend = builder.AddPnpmApp("frontend", "../semantic-search-frontned", "dev")
//     .WithReference(bff)
//     .WaitFor(bff)
//     .WithHttpEndpoint(port: 5173, targetPort: 5173, env: "PORT")
//     .WithExternalHttpEndpoints();

// =============================================================================
// Legacy Web Frontend (Aspire default)
// =============================================================================

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