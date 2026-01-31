using PgApi.Services;

var builder = WebApplication.CreateBuilder(args);

// Aspire defaults (includes AddHealthChecks with a "self" check)
builder.AddServiceDefaults();

// Add PostgreSQL data source from Aspire
builder.AddNpgsqlDataSource("HM");

// Services
builder.Services.AddGrpc();

// Register the gRPC health service (no extra .AddCheck("self"))
builder.Services.AddGrpcHealthChecks();

var app = builder.Build();

// Maps Aspire's default HTTP health endpoints
app.MapDefaultEndpoints();

// gRPC service + gRPC health service
app.MapGrpcService<GreeterService>();
app.MapGrpcService<ProductService>();
app.MapGrpcHealthChecksService();

app.MapGet("/",
    () =>
        "Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");

app.Run();