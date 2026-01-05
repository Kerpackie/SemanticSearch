using MnemeTester;

var builder = WebApplication.CreateBuilder(args);

builder.AddServiceDefaults();

// OpenAPI (optional)
builder.Services.AddOpenApi();

// ---- gRPC client registration (Aspire service discovery) ----
// Use http:// if mneme-api is plaintext (h2c). Use https:// if it serves TLS.
builder.Services.AddGrpcClient<ProductSearch.ProductSearchClient>(o =>
    {
        o.Address = new Uri("http://mneme-api");
    })
    .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler())
    .ConfigureChannel(_ =>
    {
        // Needed for HTTP/2 over plaintext (h2c)
        AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);
    });

var app = builder.Build();

app.MapDefaultEndpoints();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

// ---- REST -> gRPC bridge endpoint ----
app.MapPost("/products/search", async (ProductSearch.ProductSearchClient client, SearchDto dto) =>
    {
        var req = new ProductSearchRequest
        {
            Limit = dto.Limit
        };
        if (dto.TextVector is not null)
            req.TextVector.AddRange(dto.TextVector);
        
        if (dto.ImageVector is not null)
            req.ImageVector.AddRange(dto.ImageVector);

        var reply = await client.SearchProductsAsync(req);

        // Shape the response nicely for HTTP callers
        var result = reply.Products.Select(p => new
        {
            p.Id,
            p.Name,
            p.Description,
            p.Score
        });

        return Results.Ok(result);
    })
    .WithName("SearchProducts");


app.Run();

// Simple request DTO for the HTTP endpoint
public sealed record SearchDto(float[] TextVector, float[] ImageVector, int Limit);